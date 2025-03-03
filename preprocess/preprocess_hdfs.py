import ast
import os
import re
import gc
import glob
import psutil  # 添加内存监控
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.append('/workspace/2025/LogFormer') 
import Drain

log_name = 'HDFS'
input_dir = 'log_data/'  # The input directory of log file
output_dir = 'parse_result/'  # The output directory of parsing results


def check_memory():
    """检查内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"Current memory usage: {memory_gb:.2f} GB")
    return memory_gb


def preprocess_data(df, mode, batch_size=1000):
    """分批处理数据以减少内存使用"""
    x_data, y_data = [], []
    unique_blocks = df['BlockId'].unique()
    total_blocks = len(unique_blocks)
    pbar = tqdm(total=total_blocks, desc=f'{mode} data collection')
    
    # 分批处理
    for i in range(0, total_blocks, batch_size):
        batch_blocks = unique_blocks[i:i+batch_size]
        batch_df = df[df['BlockId'].isin(batch_blocks)]
        
        # 处理当前批次
        for blk_id in batch_blocks:
            blk_data = batch_df[batch_df['BlockId'] == blk_id]
            if len(blk_data) > 0:
                x_data.append(np.array(blk_data['Vector'].tolist()))
                y_index = int(blk_data.iloc[0]['Label'] == 'Anomaly')
                y = [0, 0]
                y[y_index] = 1
                y_data.append(y)
                pbar.update(1)
        
        # 当累积一定数量的数据后保存并清理
        if len(x_data) >= batch_size:
            # 保存当前批次
            temp_file = f'preprocessed_data/{log_name}_{mode}_temp_{i}.npz'
            np.savez(temp_file, 
                    x=np.array(x_data, dtype=object),
                    y=np.array(y_data))
            x_data = []
            y_data = []
            # 清理内存
            del batch_df
            gc.collect()
            check_memory()
    
    # 保存最后一批
    if x_data:
        temp_file = f'preprocessed_data/{log_name}_{mode}_temp_final.npz'
        np.savez(temp_file,
                x=np.array(x_data, dtype=object),
                y=np.array(y_data))
    
    pbar.close()
    
    # 合并所有临时文件
    merge_temp_files(mode)

def merge_temp_files(mode):
    """合并临时文件，将测试集分成两部分保存"""
    print(f'Merging {mode} files...')
    all_x = []
    all_y = []
    
    temp_files = sorted(glob.glob(f'preprocessed_data/{log_name}_{mode}_temp_*.npz'))
    total_files = len(temp_files)
    
    if mode == 'testing':
        # 测试集分两部分处理
        mid_point = total_files // 2
        
        # 处理第一部分
        print("Processing first half of testing data...")
        for temp_file in tqdm(temp_files[:mid_point], desc='Merging files (part 1)'):
            data = np.load(temp_file, allow_pickle=True)
            all_x.extend(data['x'])
            all_y.extend(data['y'])
            del data
            gc.collect()
            os.remove(temp_file)
        
        # 保存第一部分
        np.savez(f'preprocessed_data/{log_name}_{mode}.npz',
                 x=np.array(all_x, dtype=object),
                 y=np.array(all_y))
        del all_x, all_y
        gc.collect()
        
        # 处理第二部分
        print("Processing second half of testing data...")
        all_x = []
        all_y = []
        for temp_file in tqdm(temp_files[mid_point:], desc='Merging files (part 2)'):
            data = np.load(temp_file, allow_pickle=True)
            all_x.extend(data['x'])
            all_y.extend(data['y'])
            del data
            gc.collect()
            os.remove(temp_file)
        
        # 保存第二部分
        np.savez(f'preprocessed_data/{log_name}_{mode}_1.npz',
                 x=np.array(all_x, dtype=object),
                 y=np.array(all_y))
    else:
        # 训练集正常处理
        for temp_file in tqdm(temp_files, desc='Merging files'):
            data = np.load(temp_file, allow_pickle=True)
            all_x.extend(data['x'])
            all_y.extend(data['y'])
            del data
            gc.collect()
            os.remove(temp_file)
        
        # 保存结果
        np.savez(f'preprocessed_data/{log_name}_{mode}.npz',
                 x=np.array(all_x, dtype=object),
                 y=np.array(all_y))


if __name__ == '__main__':
    if not os.path.exists(output_dir+log_name+'.log_structured.csv'):
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
        # Regular expression list for optional preprocessing (default: [])
        regex = [
            r'blk_(|-)[0-9]+',  # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes

        parser = Drain.LogParser(log_format, indir=input_dir,
                                 outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(log_name+'.log')

    print('Initial memory usage:')
    check_memory()

    num_workers = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        'distilbert-base-nli-mean-tokens', device=device)

    structured_file_name = log_name+'.log_structured.csv'
    template_file_name = log_name+'.log_templates.csv'

    # load data
    df_template = pd.read_csv(output_dir + template_file_name)
    df_structured = pd.read_csv(output_dir + structured_file_name)
    df_label = pd.read_csv(input_dir+'preprocessed/anomaly_label.csv')

    print('After loading data:')
    check_memory()

    # calculate vectors for all known templates
    print('vector embedding...')
    with torch.no_grad():  # 添加这行来减少内存使用
        embeddings = model.encode(
            df_template['EventTemplate'].tolist())
    df_template['Vector'] = list(embeddings)
    template_dict = df_template.set_index('EventTemplate')['Vector'].to_dict()
    del df_template, embeddings
    gc.collect()

    print('After template processing:')
    check_memory()

    # convert templates to vectors for all logs
    vectors = []
    for idx, template in enumerate(df_structured['EventTemplate']):
        if idx % 1000 == 0:  # 定期检查内存
            if check_memory() > 35:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        try:
            vectors.append(template_dict[template])
        except KeyError:
            with torch.no_grad():
                vectors.append(model.encode(template))
    df_structured['Vector'] = vectors
    del vectors
    gc.collect()
    print('done')

    # remove unused column
    df_structured.drop(columns=['Date', 'Time', 'Pid', 'Level', 'Component',
                                'Content', 'EventId', 'EventTemplate'], axis=1, inplace=True)

    # extract BlockId
    r1 = re.compile('^blk_-?[0-9]')
    r2 = re.compile('.*blk_-?[0-9]')

    paramlists = df_structured['ParameterList'].tolist()
    blk_id_list = []
    for paramlist in tqdm(paramlists, desc='extract BlockId'):
        paramlist = ast.literal_eval(paramlist)
        blk_id = list(filter(r1.match, paramlist))

        if len(blk_id) == 0:
            filter_str_list = list(filter(r2.match, paramlist))
            # ex: '/mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906'
            blk_id = filter_str_list[0].split(' ')[-1]
        else:
            # ex: ['blk_-1608999687919862906'], ['blk_-1608999687919862906', 'blk_-1608999687919862906'],
            # ['blk_-1608999687919862906 terminating']
            blk_id = blk_id[0].split(' ')[0]

        blk_id_list.append(blk_id)

    df_structured['BlockId'] = blk_id_list
    df_structured.drop(columns=['ParameterList'], axis=1, inplace=True)

    # 修改回原始的数据集划分方式
    print('Preparing labels...')
    df_label['Usage'] = 'testing'
    train_index = df_label.sample(frac=0.2, random_state=123).index  # 随机抽取20%作为训练集
    df_label.iloc[train_index, df_label.columns.get_loc('Usage')] = 'training'

    print('Before merging:')
    check_memory()

    df_structured = pd.merge(df_structured, df_label, on='BlockId')
    del df_label
    gc.collect()

    # group data by BlockId
    df_structured.sort_values(by=['BlockId', 'LineId'], inplace=True)
    df_structured.drop(columns=['LineId'], axis=1, inplace=True)

    print('Before splitting:')
    check_memory()

    # split training and testing dataframe
    df_test = df_structured[df_structured['Usage'] == 'testing']
    df_train = df_structured[df_structured['Usage'] == 'training']
    del df_structured
    gc.collect()

    print('Processing training data:')
    check_memory()
    preprocess_data(df_train, 'training')
    del df_train
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print('Processing testing data:')
    check_memory()
    preprocess_data(df_test, 'testing')
    del df_test
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print('Final memory usage:')
    check_memory()
