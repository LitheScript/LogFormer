from logging import raiseExceptions
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import re
import torch
from sentence_transformers import SentenceTransformer
import gc
import os

def batch_encode_templates(model, templates, batch_size=16):
    """批量编码模板，使用更小的batch_size"""
    vectors = []
    for i in tqdm(range(0, len(templates), batch_size)):
        batch = templates[i:i+batch_size]
        with torch.no_grad():
            batch_vectors = model.encode(batch, show_progress_bar=False)
        vectors.extend(batch_vectors)
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    return vectors

def process_chunk(df_chunk, template_dict, model, chunk_idx, mode):
    """处理单个数据块并保存中间结果"""
    vectors = []
    for template in tqdm(df_chunk['EventTemplate'], 
                        desc=f"Processing chunk {chunk_idx}",
                        leave=False):
        try:
            vectors.append(template_dict[template])
        except KeyError:
            with torch.no_grad():
                vectors.append(model.encode(template))
    
    df_chunk['Vector'] = vectors
    
    temp_file = f'temp_{mode}_chunk_{chunk_idx}.npz'
    x_data, y_data = [], []
    
    for i in range(0, len(df_chunk), 20):
        if i + 20 <= len(df_chunk):
            df_blk = df_chunk.iloc[i:i+20]
            x_data.append(np.array(df_blk["Vector"].tolist()))
            labels = df_blk["Label"].tolist()
            y_data.append([1, 0] if labels == ['-']*20 else [0, 1])
    
    if x_data:
        np.savez(temp_file, x=np.array(x_data), y=np.array(y_data))
    
    del vectors, x_data, y_data, df_chunk
    gc.collect()
    return temp_file

def merge_temp_files(temp_files, mode, log_name):
    """合并临时文件"""
    all_x, all_y = [], []
    
    for temp_file in temp_files:
        data = np.load(temp_file)
        all_x.extend(data['x'])
        all_y.extend(data['y'])
        os.remove(temp_file)
    
    # 保持原始文件命名
    np.savez(f'preprocessed_data/{log_name}_{mode}.npz',
             x=np.array(all_x), y=np.array(all_y))
    
    del all_x, all_y
    gc.collect()

if __name__ == '__main__':
    # 设置参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    chunk_size = 500
    log_name = 'BGL'
    
    # 创建临时目录
    if not os.path.exists('preprocessed_data'):
        os.makedirs('preprocessed_data')
    
    # 使用原始模型
    print('Loading model...')
    model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)
    
    # 处理模板
    print('Processing templates...')
    df_template = pd.read_csv(f"parse_result/{log_name}.log_templates.csv")
    embeddings = batch_encode_templates(model, df_template['EventTemplate'].tolist(), batch_size)
    template_dict = dict(zip(df_template['EventTemplate'], embeddings))
    del df_template, embeddings
    gc.collect()
    
    # 分块读取和处理结构化日志
    print('Processing structured logs...')
    chunks = pd.read_csv(f"parse_result/{log_name}.log_structured.csv", 
                        chunksize=chunk_size)
    
    temp_files_train = []
    temp_files_test = []
    total_rows = 0
    
    # 计算总行数用于划分训练集和测试集
    for chunk in pd.read_csv(f"parse_result/{log_name}.log_structured.csv", 
                           chunksize=chunk_size):
        total_rows += len(chunk)
    
    training_rows = (total_rows//20//5)*4*20  # 保持原始的80/20分割
    current_rows = 0
    chunk_idx = 0
    
    # 重新读取并处理数据
    chunks = pd.read_csv(f"parse_result/{log_name}.log_structured.csv", 
                        chunksize=chunk_size)
    
    for chunk in chunks:
        # 删除不需要的列
        columns_to_drop = ['Date', 'Node', 'Time', 'NodeRepeat', 'Type', 
                          'Component', 'Level']
        existing_columns = [col for col in columns_to_drop if col in chunk.columns]
        if existing_columns:
            chunk = chunk.drop(columns=existing_columns)
        
        # 确定当前chunk是训练集还是测试集
        if current_rows < training_rows:
            mode = 'training'
            temp_files_train.append(process_chunk(chunk, template_dict, model, 
                                                chunk_idx, mode))
        else:
            mode = 'testing'
            temp_files_test.append(process_chunk(chunk, template_dict, model, 
                                               chunk_idx, mode))
        
        current_rows += len(chunk)
        chunk_idx += 1
        gc.collect()
    
    # 合并临时文件
    print('Merging training files...')
    merge_temp_files(temp_files_train, 'training', log_name)
    print('Merging testing files...')
    merge_temp_files(temp_files_test, 'testing', log_name)
    
    # 清理最终内存
    del template_dict, model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
