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
input_dir = 'log_data/preprocessed/'  # The input directory of log file
output_dir = 'parse_result/'  # The output directory of parsing results


def check_memory():
    """检查内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"Current memory usage: {memory_gb:.2f} GB")
    return memory_gb


def preprocess_data(df, mode):
    """处理数据并保存为最终文件"""
    x_template_data = []
    x_param_data = []
    y_data = []
    pbar = tqdm(total=df['BlockId'].nunique(),
                desc=f'{mode} data collection')

    while len(df) > 0:
        blk_id = df.iloc[0]['BlockId']
        last_index = 0
        for i in range(len(df)):
            if df.iloc[i]['BlockId'] != blk_id:
                break
            last_index += 1

        df_blk = df[:last_index]
        x_template_data.append(df_blk['TemplateVector'].tolist())
        x_param_data.append(df_blk['ParamVector'].tolist())

        y_index = int(df_blk.iloc[0]['Label'] == 'Anomaly')
        y = [0, 0]
        y[y_index] = 1
        y_data.append(y)

        df = df.iloc[last_index:]
        pbar.update()
    pbar.close()

    print(f'Saving {mode} data...')
    np.savez(f'preprocessed_data/{log_name}_{mode}_param_attn_test{output_suffix}.npz',
             x_template=np.array(x_template_data, dtype=object),
             x_param=np.array(x_param_data, dtype=object),
             y=np.array(y_data))
    
    print(f'Saved {len(x_template_data)} sequences')
    print(f'Example sequence length: {len(x_template_data[0])}')


def batch_process_parameters(params_list, batch_size=1000):
    """批量处理参数列表
    Args:
        params_list: ParameterList列表
        batch_size: 批处理大小
    Returns:
        processed_texts: 处理后的文本列表
    """
    processed_texts = []
    for i in tqdm(range(0, len(params_list), batch_size), desc='Processing parameters'):
        batch = params_list[i:i + batch_size]
        # 批量处理ast.literal_eval
        batch_processed = [' '.join(ast.literal_eval(params)) for params in batch]
        processed_texts.extend(batch_processed)
    return processed_texts


def process_param_to_chars(param_str):
    """将参数字符串转换为字符序列列表
    Args:
        param_str: 参数字符串，如'blk_-1608999687919862906 /10.250.19.102:54106'
    Returns:
        param_chars_list: 每个参数的字符序列列表
    """
    # 按空格分割参数
    params = param_str.strip().split()
    # 对每个参数进行字符级处理
    param_chars_list = []
    for param in params:
        chars = ' '.join(list(param))
        param_chars_list.append(chars)
    return param_chars_list


def batch_encode_parameters(params_list, model, batch_size=32):
    """批量编码参数列表
    Args:
        params_list: 原始参数列表
        model: BERT模型
        batch_size: 批处理大小
    Returns:
        encoded_params_list: 每个日志的参数编码列表，每个参数一个向量
    """
    encoded_params_list = []
    
    for i in tqdm(range(0, len(params_list), batch_size), desc='Encoding parameters'):
        batch = params_list[i:i + batch_size]
        batch_encoded_params = []
        
        for params in batch:
            # 处理参数字符串
            param_chars_list = process_param_to_chars(params)
            # 对每个参数单独编码
            param_encodings = []
            for char_seq in param_chars_list:
                try:
                    # 编码单个参数
                    encoding = model.encode(char_seq,
                                         show_progress_bar=False,
                                         convert_to_numpy=True)
                    param_encodings.append(encoding)
                except Exception as e:
                    print(f"Error encoding parameter {char_seq}: {e}")
                    param_encodings.append(np.zeros(768))
            
            batch_encoded_params.append(param_encodings)
        
        encoded_params_list.extend(batch_encoded_params)
    
    return encoded_params_list


if __name__ == '__main__':
    test_size = 60000  # 测试数据大小
    output_suffix = f'_{test_size}'  # 文件后缀

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)

    structured_file_name = log_name+'.log_structured.csv'
    template_file_name = log_name+'.log_templates.csv'

    # [修改] 只读取前30000行数据
    df_structured = pd.read_csv(output_dir + structured_file_name, nrows=test_size)
    df_template = pd.read_csv(output_dir + template_file_name)
    df_label = pd.read_csv(input_dir+'anomaly_label.csv')

    print('After loading data:')
    check_memory()

    # 计算模板向量
    print('Template vector embedding...')
    embeddings = model.encode(df_template['EventTemplate'].tolist())
    df_template['TemplateVector'] = list(embeddings)
    template_dict = df_template.set_index('EventTemplate')['TemplateVector'].to_dict()

    # 转换模板为向量
    template_vectors = []
    for idx, template in enumerate(df_structured['EventTemplate']):
        try:
            template_vectors.append(template_dict[template])
        except KeyError:
            template_vectors.append(model.encode(template))
    df_structured['TemplateVector'] = template_vectors

    # [优化] 参数向量编码
    print('Parameter vector embedding...')
    # 1. 批量预处理参数
    param_texts = batch_process_parameters(
        df_structured['ParameterList'].tolist(), 
        batch_size=1000
    )
    
    # 2. 分批编码
    print('Encoding parameters...')
    param_vectors = batch_encode_parameters(
        param_texts,
        model,
        batch_size=32
    )
    
    df_structured['ParamVector'] = param_vectors
    del param_texts, param_vectors  # 及时释放内存
    gc.collect()
    
    print('Parameter encoding done')
    check_memory()

    # 移除不需要的列
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
