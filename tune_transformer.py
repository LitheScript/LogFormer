# ===== 导入必要的库 =====
import argparse  # 命令行参数解析
import random   # 随机数生成
import warnings # 警告控制
import time     # 时间计算

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

# ===== 参数设置 =====
parser = argparse.ArgumentParser()
# 预训练相关设置
parser.add_argument('--pretrained_log_name', type=str,
                    default='BGL', help='预训练模型的日志数据集名称')
parser.add_argument("--load_path", type=str,
                    default='checkpoints/train_BGL_classifier_1_64_1e-05-best.pt', 
                    help="预训练模型加载路径")
# 目标数据集设置                    
parser.add_argument('--log_name', type=str,
                    default='HDFS', help='目标数据集名称')
# 微调模式设置                    
parser.add_argument('--tune_mode', type=str, default='adapter',
                    help='微调模式: adapter/classifier/tuning')

# 模型架构设置
parser.add_argument('--num_layers', type=int, default=1,
                    help='Transformer编码器层数')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='学习率')
parser.add_argument('--window_size', type=int,
                    default='20', help='日志序列长度')
parser.add_argument('--adapter_size', type=int, default=64,
                    help='Adapter层大小')
parser.add_argument('--epoch', type=int, default=20,
                    help='训练轮数')

# 解析参数
args = parser.parse_args()
# 生成结果文件的后缀名,包含数据集转移信息和模型配置
suffix = f'{args.log_name}_from_{args.pretrained_log_name}_{args.tune_mode}_{args.num_layers}_{args.adapter_size}_{args.lr}_{args.epoch}'

# 创建结果文件并写入参数配置
with open(f'result/tune_{suffix}.txt', 'w', encoding='utf-8') as f:
    f.write(str(args)+'\n')

# ===== 超参数设置 =====
EMBEDDING_DIM = 768  # 词嵌入维度
batch_size = 64     # 批次大小
epochs = args.epoch  # 训练轮数
# 设置计算设备(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0] # 多GPU训练设置

# ===== 固定随机种子 =====
warnings.filterwarnings('ignore')  # 忽略警告信息
torch.manual_seed(123)            # 设置PyTorch的随机种子
torch.cuda.manual_seed(123)       # 设置CUDA的随机种子
np.random.seed(123)              # 设置NumPy的随机种子
random.seed(123)                 # 设置Python的随机种子
torch.backends.cudnn.deterministic = True  # 确保CUDNN的确定性

# ===== 数据加载 =====
# 加载训练数据
training_data = np.load(
    f'./preprocessed_data/{args.log_name}_training.npz', allow_pickle=True)
# 加载测试数据
testing_data = np.load(
    f'./preprocessed_data/{args.log_name}_testing.npz', allow_pickle=True)
x_train, y_train = training_data['x'], training_data['y']  # 提取训练数据和标签
x_test, y_test = testing_data['x'], testing_data['y']      # 提取测试数据和标签
del testing_data  # 释放内存
del training_data

# ===== 创建数据加载器 =====
# 创建训练数据生成器
train_generator = DataGenerator(x_train, y_train, args.window_size)
# 创建测试数据生成器
test_generator = DataGenerator(x_test, y_test, args.window_size)
# 创建训练数据加载器
train_loader = torch.utils.data.DataLoader(
    train_generator, batch_size=batch_size, shuffle=True)
# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(
    test_generator, batch_size=batch_size, shuffle=False)

# ===== 模型初始化 =====
# 创建模型实例
model = Model(mode='adapter', num_layers=args.num_layers, adapter_size=args.adapter_size, 
             dim=EMBEDDING_DIM, window_size=args.window_size, nhead=8, 
             dim_feedforward=4*EMBEDDING_DIM, dropout=0.1)

# ===== 设置微调模式 =====
if args.tune_mode == 'adapter':
    model.train_adapter()  # 只训练adapter层
elif args.tune_mode == 'classifier':
    model.train_classifier()  # 只训练分类器层
elif args.tune_mode == 'tuning':
    for param in model.parameters():  # 训练所有参数
        param.requires_grad = True

# 将模型移至GPU并设置多GPU训练
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=device_ids)

# ===== 加载预训练模型 =====
if args.pretrained_log_name != 'random':  # 如果不是随机初始化
    checkpoint = torch.load(args.load_path)  # 加载预训练模型
    net = checkpoint['net']
    # 移除分类器层的参数(将重新初始化)
    net.pop('module.fc1.weight')
    net.pop('module.fc1.bias')
    # 加载模型参数
    r = model.load_state_dict(net, strict=False)
    # 记录加载结果
    with open(f'result/tune_{suffix}.txt', 'a', encoding='utf-8') as f:
        f.write(f'loading pretrained model {args.load_path}\n')
        f.write(f'loading result: {r}\n')

# ===== 优化器和学习率调度器设置 =====
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)  # Adam优化器
# 使用OneCycleLR学习率调度器
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr, epochs=epochs, steps_per_epoch=len(train_loader))
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失

# ===== 初始化训练指标 =====
best_f1 = 0  # 最佳F1分数
start_epoch = -1  # 起始轮数
log_interval = 100  # 日志打印间隔

# ===== 训练循环 =====
for epoch in range(start_epoch + 1, epochs):
    # 初始化训练指标
    loss_all, f1_all = [], []  # 存储所有batch的损失和F1分数
    train_loss = 0            # 当前训练损失
    train_pred, train_true = [], []  # 存储预测结果和真实标签

    # 切换到训练模式
    model.train()
    start_time = time.time()  # 记录开始时间
    
    # 遍历训练数据批次
    for batch_idx, data in enumerate(tqdm(train_loader)):
        # 准备数据
        x, y = data[0].to(device), data[1].to(device)  # 将数据移至GPU
        x = x.to(torch.float32)  # 转换输入数据类型
        y = y.to(torch.float32)  # 转换标签数据类型
        
        # 前向传播
        out = model(x)  # 模型预测
        loss = criterion(out, y)  # 计算损失
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
        optimizer.step()       # 更新参数
        scheduler.step()       # 更新学习率
        
        # 记录训练指标
        train_loss += loss.item()  # 累积损失
        train_pred.extend(out.argmax(1).tolist())  # 记录预测结果
        train_true.extend(y.argmax(1).tolist())    # 记录真实标签
        
        # 定期打印训练信息
        if batch_idx % log_interval == 0 and batch_idx > 0:
            time_cost = time.time()-start_time  # 计算耗时
            cur_loss = train_loss / log_interval  # 计算平均损失
            cur_f1 = f1_score(train_true, train_pred)  # 计算F1分数
            
            # 写入训练日志
            with open(f'result/tune_{suffix}.txt', 'a', encoding='utf-8') as f:
                f.write(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                        f'loss {cur_loss:2.5f} |'
                        f'f1 {cur_f1:.5f} |'
                        f'time {time_cost:4.2f} |'
                        f'lr {scheduler.get_last_lr()}\n')
            
            # 打印训练信息
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                  f'loss {cur_loss} |'
                  f'f1 {cur_f1}',
                  f'lr {scheduler.get_last_lr()}')
            
            # 更新指标列表
            loss_all.append(train_loss)
            f1_all.append(cur_f1)
            
            # 重置计数器
            train_loss = 0
            train_acc = 0
            start_time = time.time()

    # 计算epoch平均损失
    train_loss = sum(loss_all) / len(train_loader)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch, epochs, train_loss))

    # ===== 评估阶段 =====
    model.eval()  # 切换到评估模式
    n = 0.0
    
    # 在测试集上评估
    with torch.no_grad():  # 不计算梯度
        for batch_idx, data in enumerate(tqdm(test_loader)):
            # 准备数据
            x, y = data[0].to(device), data[1].to(device)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            out = model(x).cpu()  # 模型预测
            
            # 收集预测结果
            if batch_idx == 0:
                y_pred = out
                y_true = y.cpu()
            else:
                y_pred = np.concatenate((y_pred, out), axis=0)
                y_true = np.concatenate((y_true, y.cpu()), axis=0)

    # 计算评估指标
    y_true = np.argmax(y_true, axis=1)  # 转换为类别索引
    y_pred = np.argmax(y_pred, axis=1)
    report = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # 写入评估结果
    with open(f'result/tune_{suffix}.txt', 'a', encoding='utf-8') as f:
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
