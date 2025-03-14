import numpy as np

# 加载预处理后的数据
data = np.load('preprocessed_data/HDFS_training_param_attn_test_60000.npz', allow_pickle=True)

# 打印数据键
print("数据键:", data.files)

# 选择5条数据进行展示
num_samples = 5
x_template_samples = data['x_template'][:num_samples]
x_param_samples = data['x_param'][:num_samples]
y_samples = data['y'][:num_samples]

# 打印每条数据的详细信息
for i in range(num_samples):
    print(f"\n示例 {i+1}:")
    print("模板向量 (x_template):")
    template_array = np.array(x_template_samples[i])  # 转换为NumPy数组
    print("Shape:", template_array.shape)
    print("Data:", template_array)

    print("\n参数向量 (x_param):")
    print("Number of parameters:", len(x_param_samples[i]))
    for j, param_vector in enumerate(x_param_samples[i]):
        param_array = np.array(param_vector)  # 转换为NumPy数组
        print(f"参数 {j+1} shape:", param_array.shape)
        print(f"参数 {j+1} data:", param_array)

    print("\n标签 (y):")
    print("Data:", y_samples[i])