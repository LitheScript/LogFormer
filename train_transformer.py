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
# 日志数据集名称,默认BGL
parser.add_argument('--log_name', type=str,
                    default='HDFS', help='log file name')
# 序列窗口大小,默认20
parser.add_argument('--window_size', type=int,
                    default='20', help='log sequence length')
# 模型模式,是否使用adapter,默认classifier
parser.add_argument('--mode', type=str, default='classifier',
                    help='use adapter or not')
# Transformer编码器层数,默认1层
parser.add_argument('--num_layers', type=int, default=1,
                    help='num of encoder layer')
# Adapter层的隐藏层大小,默认64
parser.add_argument('--adapter_size', type=int, default=64,
                    help='adapter size')
# 学习率,默认0.00001
parser.add_argument('--lr', type=float, default=0.00001)
# 是否从检查点恢复训练,默认0(否)
parser.add_argument("--resume", type=int, default=0,
                    help="resume training of model (0/no, 1/yes)")
# 加载模型的路径
parser.add_argument("--load_path", type=str,
                    default='checkpoints/model-latest.pt', help="latest model path")

# 解析参数
args = parser.parse_args()
# 生成结果文件的后缀名,包含数据集、模型模式、层数等信息
suffix = f'{args.log_name}_{args.mode}_{args.num_layers}_{args.adapter_size}_{args.lr}'
# 创建结果文件并写入参数配置
with open(f'result/train_{suffix}.txt', 'a', encoding='utf-8') as f:
    f.write(str(args)+'\n')

# 设置超参数
EMBEDDING_DIM = 768  # 词嵌入维度
batch_size = 64     # 批次大小
epochs = 30          # 训练轮数
lr = args.lr        # 学习率

# 设置计算设备(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0] # 多GPU训练设置
print('Using device = ', device)
print(f'Model mode is {args.mode}')

# 固定随机种子,确保结果可复现
warnings.filterwarnings('ignore')  # 忽略警告信息
torch.manual_seed(123)            # 设置PyTorch的随机种子
torch.cuda.manual_seed(123)       # 设置CUDA的随机种子
np.random.seed(123)              # 设置NumPy的随机种子
random.seed(123)                 # 设置Python的随机种子
torch.backends.cudnn.deterministic = True  # 确保CUDNN的确定性

# 加载训练和测试数据
training_data = np.load(
    f'./preprocessed_data/{args.log_name}_training.npz', allow_pickle=True)
testing_data = np.load(
    f'./preprocessed_data/{args.log_name}_testing.npz', allow_pickle=True)
x_train, y_train = training_data['x'], training_data['y']  # 提取训练数据和标签
x_test, y_test = testing_data['x'], testing_data['y']      # 提取测试数据和标签
del testing_data  # 释放内存
del training_data

# 创建数据生成器和加载器
train_generator = DataGenerator(x_train, y_train, args.window_size)  # 训练数据生成器
test_generator = DataGenerator(x_test, y_test, args.window_size)     # 测试数据生成器
# 创建训练数据加载器,启用随机打乱
train_loader = torch.utils.data.DataLoader(
    train_generator, batch_size=batch_size, shuffle=True)
# 创建测试数据加载器,不打乱顺序
test_loader = torch.utils.data.DataLoader(
    test_generator, batch_size=batch_size, shuffle=False)

# 初始化模型
model = Model(mode=args.mode,                # 模型模式(adapter/classifier)
             num_layers=args.num_layers,     # Transformer编码器层数
             adapter_size=args.adapter_size, # Adapter层大小
             dim=EMBEDDING_DIM,             # 输入维度
             window_size=args.window_size,  # 序列窗口大小
             nhead=8,                       # 注意力头数
             dim_feedforward=4*EMBEDDING_DIM, # 前馈网络维度
             dropout=0.1)                    # dropout比率
model = model.to(device)  # 将模型移至GPU
model = torch.nn.DataParallel(model, device_ids=device_ids)  # 多GPU并行训练设置

# 设置优化器、学习率调度器和损失函数
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)  # Adam优化器
# OneCycleLR学习率调度器,在训练过程中动态调整学习率
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数

# 如果需要从检查点恢复训练
start_epoch = -1
if args.resume == 1:
    path_checkpoint = args.load_path  # 加载检查点路径
    checkpoint = torch.load(path_checkpoint)  # 加载检查点
    model.load_state_dict(checkpoint['net'])  # 加载模型状态
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
    start_epoch = checkpoint['epoch']  # 获取已训练的轮数
    print("resume training from epoch ", start_epoch)

# 初始化最佳指标
best_acc = np.inf  # 最佳准确率
best_f1 = 0       # 最佳F1分数
log_interval = 100  # 日志打印间隔

# 开始训练循环
for epoch in range(start_epoch+1, epochs):
    # 初始化训练指标
    loss_all, f1_all = [], []  # 存储每个batch的损失和F1分数
    train_loss = 0            # 累积训练损失
    train_pred, train_true = [], []  # 存储预测值和真实值
    
    # 切换到训练模式
    model.train()
    start_time = time.time()  # 记录开始时间
    
    # 批次训练循环
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
        loss.backward()        # 反向传播
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
        optimizer.step()       # 更新参数
        scheduler.step()       # 更新学习率
        
        # 记录训练指标
        train_loss += loss.item()  # 累积损失
        train_pred.extend(out.argmax(1).tolist())  # 记录预测结果
        train_true.extend(y.argmax(1).tolist())    # 记录真实标签
        
        # 定期打印训练信息
        if batch_idx % log_interval == 0 and batch_idx > 0:
            cur_loss = train_loss / log_interval  # 计算平均损失
            cur_f1 = f1_score(train_true, train_pred)  # 计算F1分数
            time_cost = time.time()-start_time  # 计算耗时
            
            # 写入训练日志
            with open(f'result/train_{suffix}.txt', 'a', encoding='utf-8') as f:
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
            start_time = time.time()
            train_loss = 0
            train_acc = 0
    
    # 计算epoch平均损失
    train_loss = sum(loss_all) / len(train_loader)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch, epochs, train_loss))
    
    # 切换到评估模式
    model.eval()
    n = 0.0
    acc = 0.0
    
    # 在测试集上评估
    with torch.no_grad():  # 不计算梯度
        for batch_idx, data in enumerate(tqdm(test_loader)):
            # 准备数据
            x, y = data[0].to(device), data[1].to(device)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            out = model(x).cpu()  # 模型预测并移至CPU
            
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
    # 计算精确率、召回率、F1分数
    report = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # 写入评估结果
    with open(f'result/train_{suffix}.txt', 'a', encoding='utf-8') as f:
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
    
    # 保存模型检查点
    ckpt_path = 'checkpoints/'  # 检查点保存路径
    checkpoint = {
        "net": model.state_dict(),         # 保存模型参数
        'optimizer': optimizer.state_dict(),  # 保存优化器状态
        "epoch": epoch                      # 保存当前轮数
    }
    # 如果当前F1分数更好,保存最佳模型
    if report[2] > best_f1:
        best_f1 = report[2]
        torch.save(checkpoint, os.path.join(
            ckpt_path, f'train_{suffix}-best.pt'))
    # 保存最新模型
    torch.save(checkpoint, os.path.join(
        ckpt_path, f'train_{suffix}-latest.pt'))
