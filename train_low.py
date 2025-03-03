# 导入必要的库
import argparse  # 命令行参数解析
import time     # 时间计算
import os       # 文件操作
import random   # 随机数生成
import warnings # 警告控制

# 导入科学计算和深度学习相关库
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tqdm import tqdm

# 导入自定义模块
from dataloader import DataGenerator  # 数据加载器
from model import Model              # 模型定义

# 设置命令行参数
parser = argparse.ArgumentParser()
# 日志数据集名称,默认HDFS
parser.add_argument('--log_name', type=str, default='BGL', help='log file name')
# 序列窗口大小,默认50
parser.add_argument('--window_size', type=int, default='50', help='log sequence length')
# 模型模式,是否使用adapter,默认classifier
parser.add_argument('--mode', type=str, default='classifier', help='use adapter or not')
# Transformer编码器层数,默认1层
parser.add_argument('--num_layers', type=int, default=1, help='num of encoder layer')
# Adapter层的隐藏层大小,默认64
parser.add_argument('--adapter_size', type=int, default=64, help='adapter size')
# 学习率,默认5e-5
parser.add_argument('--lr', type=float, default=5e-5)
# 是否从检查点恢复训练,默认0(否)
parser.add_argument("--resume", type=int, default=0, help="resume training of model (0/no, 1/yes)")
# 加载模型的路径
parser.add_argument("--load_path", type=str, default='checkpoints/model-latest.pt', help="latest model path")
# 训练样本数量,默认50000
parser.add_argument("--num_samples", type=int, default='50000', help="number of training samples")

# 解析参数
args = parser.parse_args()
# 生成结果文件的后缀名
suffix = f'{args.log_name}_{args.mode}_{args.num_layers}_{args.adapter_size}_{args.lr}'

# 创建结果文件并写入参数配置
with open(f'result_{args.num_samples}/train_{suffix}.txt', 'w', encoding='utf-8') as f:
    f.write(str(args)+'\n')

# 设置超参数
EMBEDDING_DIM = 768  # 词嵌入维度
batch_size = 64     # 批次大小
epochs = 10         # 训练轮数
lr = args.lr        # 学习率
# 设置计算设备(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0] # 多GPU训练设置
print('Using device = ', device)
print(f'Model mode is {args.mode}')

# 固定随机种子,确保结果可复现
warnings.filterwarnings('ignore')
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True

# 加载训练和测试数据
training_data = np.load(f'./preprocessed_data/{args.log_name}_training.npz', allow_pickle=True)
testing_data = np.load(f'./preprocessed_data/{args.log_name}_testing.npz', allow_pickle=True)
x_train, y_train = training_data['x'], training_data['y']
x_test, y_test = testing_data['x'], testing_data['y']
del testing_data
del training_data

# 创建数据生成器和加载器
train_generator = DataGenerator(x_train[:args.num_samples], y_train[:args.num_samples], args.window_size)
test_generator = DataGenerator(x_test, y_test, args.window_size)
train_loader = torch.utils.data.DataLoader(train_generator, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_generator, batch_size=batch_size, shuffle=False)

# 初始化模型
model = Model(mode=args.mode, num_layers=args.num_layers, adapter_size=args.adapter_size, 
             dim=EMBEDDING_DIM, window_size=args.window_size, nhead=8, 
             dim_feedforward=4*EMBEDDING_DIM, dropout=0.1)
model = model.to(device)  # 将模型移至GPU
model = torch.nn.DataParallel(model, device_ids=device_ids)  # 多GPU并行

# 设置优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失

# 如果需要从检查点恢复训练
start_epoch = -1
if args.resume == 1:
    path_checkpoint = args.load_path
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print("resume training from epoch ", start_epoch)

# 初始化最佳指标
best_acc = np.inf
best_f1 = 0
log_interval = 100  # 日志打印间隔

# 开始训练循环
for epoch in range(start_epoch+1, epochs):
    # 初始化训练指标
    loss_all, f1_all = [], []
    train_loss = 0
    train_pred, train_true = [], []
    
    # 训练模式
    model.train()
    start_time = time.time()
    
    # 批次训练
    for batch_idx, data in enumerate(tqdm(train_loader)):
        # 准备数据
        x, y = data[0].to(device), data[1].to(device)
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        
        # 前向传播
        out = model(x)
        loss = criterion(out, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
        optimizer.step()
        
        # 记录训练指标
        train_loss += loss.item()
        train_pred.extend(out.argmax(1).tolist())
        train_true.extend(y.argmax(1).tolist())
        
        # 定期打印训练信息
        if batch_idx % log_interval == 0 and batch_idx > 0:
            cur_loss = train_loss / log_interval
            cur_f1 = f1_score(train_true, train_pred)
            time_cost = time.time()-start_time
            
            # 写入训练日志
            with open(f'result_{args.num_samples}/train_{suffix}.txt', 'a', encoding='utf-8') as f:
                f.write(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                        f'loss {cur_loss:2.5f} |'
                        f'f1 {cur_f1:.5f} |'
                        f'time {time_cost:4.2f}\n')
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                  f'loss {cur_loss} |'
                  f'f1 {cur_f1}')
            
            # 更新指标列表
            loss_all.append(train_loss)
            f1_all.append(cur_f1)
            
            # 重置计数器
            start_time = time.time()
            train_loss = 0
            train_acc = 0
    
    # 计算epoch平均损失
    train_loss = sum(loss_all) / len(train_loader)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch, epochs, train_loss))
    
    # 评估模式
    model.eval()
    n = 0.0
    acc = 0.0
    
    # 在测试集上评估
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            x, y = data[0].to(device), data[1].to(device)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            out = model(x).cpu()
            
            # 收集预测结果
            if batch_idx == 0:
                y_pred = out
                y_true = y.cpu()
            else:
                y_pred = np.concatenate((y_pred, out), axis=0)
                y_true = np.concatenate((y_true, y.cpu()), axis=0)
    
    # 计算评估指标
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    report = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # 写入评估结果
    with open(f'result_{args.num_samples}/train_{suffix}.txt', 'a', encoding='utf-8') as f:
        f.write('number of epochs:'+str(epoch)+'\n')
        f.write('Number of testing data:'+str(x_test.shape[0])+'\n')
        f.write('Precision:'+str(report[0])+'\n')
        f.write('Recall:'+str(report[1])+'\n')
        f.write('F1 score:'+str(report[2])+'\n')
        f.write('all_loss:'+str(loss_all)+'\n')
        f.write('\n')
        f.close()
    
    # 打印评估结果
    print(f'Number of testing data: {x_test.shape[0]}')
    print(f'Precision: {report[0]:.4f}')
    print(f'Recall: {report[1]:.4f}')
    print(f'F1 score: {report[2]:.4f}')