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

# 在文件开头定义一个全局线性层时指定dtype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
linear_layer = torch.nn.Linear(768, 768).to(device).float()  # 显式指定为float类型

def check_memory():
    """检查内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"Current memory usage: {memory_gb:.2f} GB")
    return memory_gb


def preprocess_data(df, mode):
    """处理数据并保存为最终文件
    Args:
        df: 按BlockId分组后的数据框
        mode: 'training' 或 'testing'
    """
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


def process_single_parameter(param):
    """处理单个参数
    Args:
        param: 单个参数字符串
    Returns:
        chars: 字符级别的表示
    """
    # 将参数拆分成字符
    chars = ' '.join(list(param.strip()))
    return chars


def batch_process_parameters(params_list, model, batch_size=1024):
    """批量处理参数列表，使用更大的batch_size和chunk_size来提高速度
    Args:
        params_list: 参数列表
        model: 编码模型
        batch_size: 批处理大小
    Returns:
        encoded_params_list: 编码后的参数列表
    """
    encoded_params_list = []
    model = model.to(device)
    
    # 增加chunk_size，减少循环次数
    chunk_size = 50000  # 增加到50000
    for chunk_start in tqdm(range(0, len(params_list), chunk_size), desc='Processing chunks'):
        chunk_end = min(chunk_start + chunk_size, len(params_list))
        chunk_params = params_list[chunk_start:chunk_end]
        
        all_char_sequences = []
        sequence_map = []
        
        # 预处理当前chunk的参数
        for params in chunk_params:
            if isinstance(params, str):
                params = ast.literal_eval(params)
            
            start_idx = len(all_char_sequences)
            char_sequences = [process_single_parameter(param) for param in params if param.strip()]
            
            if char_sequences:  # 只有当有有效参数时才添加
                all_char_sequences.extend(char_sequences)
                sequence_map.append((start_idx, start_idx + len(char_sequences)))
            else:
                sequence_map.append((start_idx, start_idx))  # 空参数情况
        
        # 批量编码（使用更大的batch_size）
        all_encodings = []
        for i in range(0, len(all_char_sequences), batch_size):
            batch = all_char_sequences[i:i + batch_size]
            try:
                encodings = model.encode(batch,
                                       show_progress_bar=False,
                                       convert_to_numpy=True,
                                       batch_size=batch_size,
                                       device=device)
                all_encodings.extend(encodings)
            except Exception as e:
                print(f"Error encoding batch: {e}")
                all_encodings.extend([np.zeros(768) for _ in batch])
        
        # 处理编码结果（使用列表推导式加速）
        chunk_encoded_params = [
            compress_param_vectors(all_encodings[start:end]) if start != end 
            else np.zeros(768) 
            for start, end in sequence_map
        ]
        
        encoded_params_list.extend(chunk_encoded_params)
        
        # 只在真正需要时才清理内存
        if chunk_start + chunk_size < len(params_list):  # 不是最后一个chunk
            del all_encodings, chunk_encoded_params
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return encoded_params_list


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


def compress_param_vectors(param_vectors, output_dim=768):
    """将参数向量压缩成固定维度（直接求平均）
    Args:
        param_vectors: 形状为(n, 768)的参数向量列表
        output_dim: 输出维度，默认768（此参数保留但不再使用）
    Returns:
        compressed_vector: 形状为(768,)的压缩向量
    """
    # 直接用numpy计算平均值
    return np.mean(param_vectors, axis=0)


def batch_encode_parameters(params_list, model, batch_size=32):
    """批量编码参数列表并压缩
    Args:
        params_list: 原始参数列表
        model: BERT模型
        batch_size: 批处理大小
    Returns:
        compressed_params_list: 每个日志的压缩后的参数编码向量(768维)
    """
    compressed_params_list = []
    
    for i in tqdm(range(0, len(params_list), batch_size), desc='Encoding parameters'):
        batch = params_list[i:i + batch_size]
        batch_compressed_params = []
        
        for params in batch:
            # 处理参数字符串
            param_chars_list = process_param_to_chars(params)
            # 对每个参数单独编码
            param_encodings = []
            for char_seq in param_chars_list:
                try:
                    encoding = model.encode(char_seq,
                                         show_progress_bar=False,
                                         convert_to_numpy=True)
                    param_encodings.append(encoding)
                except Exception as e:
                    print(f"Error encoding parameter {char_seq}: {e}")
                    param_encodings.append(np.zeros(768))
            
            # 压缩参数向量
            compressed = compress_param_vectors(param_encodings)
            batch_compressed_params.append(compressed)
        
        compressed_params_list.extend(batch_compressed_params)
    
    return compressed_params_list


def save_split_data(df, mode, output_suffix=''):
    """保存划分后的数据
    Args:
        df: 数据框
        mode: 'training' 或 'testing'
        output_suffix: 输出文件的后缀
    """
    x_template_data = []
    x_param_data = []
    y_data = []
    
    for blk_id in tqdm(df['BlockId'].unique(), desc=f'Processing {mode} data'):
        df_blk = df[df['BlockId'] == blk_id]
        x_template_data.append(df_blk['TemplateVector'].tolist())
        x_param_data.append(df_blk['ParamVector'].tolist())
        
        y_index = int(df_blk.iloc[0]['Label'] == 'Anomaly')
        y = [0, 0]
        y[y_index] = 1
        y_data.append(y)
    
    print(f'Saving {mode} data...')
    np.savez(f'preprocessed_data/{log_name}_{mode}_param_attn_test{output_suffix}.npz',
             x_template=np.array(x_template_data, dtype=object),
             x_param=np.array(x_param_data, dtype=object),
             y=np.array(y_data))


def process_saved_data(full_data_path, output_suffix=''):
    """从保存的完整数据中处理并保存训练集和测试集
    Args:
        full_data_path: 完整数据的.npz文件路径
        output_suffix: 输出文件的后缀
    """
    print('Loading full processed data...')
    data = np.load(full_data_path, allow_pickle=True)
    
    # 创建DataFrame，保持原有顺序
    df = pd.DataFrame({
        'BlockId': data['BlockId'],
        'TemplateVector': list(data['TemplateVector']),
        'ParamVector': list(data['ParamVector']),
        'Label': data['Label'],
        'Usage': data['Usage']
    })
    
    # 使用原始的训练/测试集划分
    print('Splitting data...')
    df_train = df[df['Usage'] == 'training']
    df_test = df[df['Usage'] == 'testing']
    
    # 保存训练集和测试集
    save_split_data(df_train, 'training', output_suffix)
    save_split_data(df_test, 'testing', output_suffix)


if __name__ == '__main__':
    # test_size = 10000000  # 测试数据大小
    output_suffix = '_with_param'  # 文件后缀

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

 
    df_structured = pd.read_csv(output_dir + structured_file_name,nrows=3000000)
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

    # extract BlockId and clean parameters
    r1 = re.compile('^blk_-?[0-9]')
    r2 = re.compile('.*blk_-?[0-9]')

    paramlists = df_structured['ParameterList'].tolist()
    blk_id_list = []
    cleaned_paramlists = []

    for paramlist in tqdm(paramlists, desc='extract BlockId and clean params'):
        paramlist = ast.literal_eval(paramlist)
        # 提取block id
        blk_id = list(filter(r1.match, paramlist))
        if len(blk_id) == 0:
            filter_str_list = list(filter(r2.match, paramlist))
            blk_id = filter_str_list[0].split(' ')[-1]
        else:
            blk_id = blk_id[0].split(' ')[0]  # 获取干净的blk_id
        
        # 统一处理：对所有参数移除blk_id
        cleaned_params = []
        for p in paramlist:
            if blk_id in p:
                cleaned_p = p.replace(blk_id, '').strip()
                if cleaned_p:  # 如果去除blk_id后还有内容，则保留
                    cleaned_params.append(cleaned_p)
            else:
                cleaned_params.append(p)
        
        blk_id_list.append(blk_id)
        cleaned_paramlists.append(cleaned_params)

    df_structured['BlockId'] = blk_id_list
    df_structured['ParameterList'] = cleaned_paramlists

    # 参数向量编码
    print('Parameter vector embedding...')
    param_vectors = batch_process_parameters(
        df_structured['ParameterList'].tolist(),
        model,
        batch_size=1024  # 或更大，取决于GPU内存
    )
    
    df_structured['ParamVector'] = param_vectors
    del param_vectors  # 及时释放内存
    gc.collect()
    
    print('Parameter encoding done')
    check_memory()

    # 完成向量化后删除不需要的列
    df_structured.drop(columns=['ParameterList'], axis=1, inplace=True)

    # 准备标签和划分
    print('Preparing labels...')
    df_label['Usage'] = 'testing'
    train_index = df_label.sample(frac=0.2, random_state=123).index
    df_label.iloc[train_index, df_label.columns.get_loc('Usage')] = 'training'

    # 合并标签
    df_structured = pd.merge(df_structured, df_label, on='BlockId')
    del df_label
    gc.collect()

    # 按BlockId和LineId排序
    print('Sorting data...')
    df_structured.sort_values(by=['BlockId', 'LineId'], inplace=True)
    df_structured.drop(columns=['LineId'], axis=1, inplace=True)

    # 直接分割并保存训练集和测试集
    print('Processing training/testing split...')
    df_train = df_structured[df_structured['Usage'] == 'training']
    df_test = df_structured[df_structured['Usage'] == 'testing']
    
    preprocess_data(df_train, 'training')
    del df_train
    gc.collect()
    
    preprocess_data(df_test, 'testing')
    del df_test
    gc.collect()

    del df_structured
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
