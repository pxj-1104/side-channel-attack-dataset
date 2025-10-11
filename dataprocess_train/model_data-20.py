import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import time

# 定义数据和标签获取函数
def get_label_from_path(file_path):
    parts = file_path.split(os.sep)
    if 'attack' in parts:
        return parts[parts.index('attack') + 1][4:]  # 将同类攻击归为一类
    elif 'normal' in parts:
        return 'normal'
    else:
        return 'unknown'


def get_load_type_from_path(file_path):
    parts = file_path.split(os.sep)
    for part in ['high', 'medium', 'low']:
        if part in parts:
            return part
    return 'unknown'

percentiles = {
    0: {'1st': 238215.0, '5th': 351578.7,'25th': 918343.0, '50th': 3046457.0, '75th': 6757895.0 ,'95th': 14359824.6, '99th': 21536046.6,  },
    1: {'1st': 1000.0, '5th': 1574.0,'25th': 7948.0, '50th': 133666.0, '75th': 634565.0 ,'95th': 3673372.3, '99th': 6137645.1,  },
    2: {'1st': 483551.0, '5th': 2123331.7,'25th': 21176662.5, '50th': 39105027.0, '75th': 63172839.5 ,'95th': 108643415.2, '99th': 145954541.9,  },
    3: {'1st': 35958.0, '5th': 54712.0,'25th': 331185.0, '50th': 1340021.0, '75th': 3030826.0 ,'95th': 6493264.6, '99th': 21240985.7,  },
    4: {'1st': 115344.0, '5th': 174028.0,'25th': 347937.0, '50th': 1295727.0, '75th': 3768945.0 ,'95th': 8629330.3, '99th': 12422065.8,  },
    5: {'1st': 13499.0, '5th': 21873.0,'25th': 84641.0, '50th': 257371.0, '75th': 647039.0 ,'95th': 1594324.3, '99th': 3484832.9,  },
    6: {'1st': 191.0, '5th': 347.0,'25th': 1123.0, '50th': 9099.0, '75th': 58920.0 ,'95th': 326293.3, '99th': 1000904.0,  },
    7: {'1st': 258.0, '5th': 527.0,'25th': 1051.0, '50th': 8030.0, '75th': 86665.0 ,'95th': 761173.0, '99th': 2551487.9,  },
}


def load_data_and_labels(base_dir):
    data = []
    file_paths = []
    labels = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, skiprows=1)  # Skip header row
                # 防止缺失值的行
                df = df.dropna()
                if len(df) > 500:
                    if 'aes_flush+reload' in file_path:
                        df = df.head(480)
                    else:
                        df = df.head(len(df) - 10)  # 舍弃掉末尾数据，因为末尾数据可能有0
                    # Add time column starting from 0.05, incrementing by 0.05
                    label = get_label_from_path(file_path)
                    num_cases = ((len(df)) // 30) - 1
                    for i in range(num_cases):  # 使用滑动窗口
                        start_row = i * 30
                        end_row = start_row + 60
                        case_data = df.iloc[start_row:end_row].values
                        # 检查最后是否包含0
                        if 0 not in case_data:
                            data.append(case_data)
                            file_paths.append(file_path)
                            labels.append(label)
    return data, file_paths, labels
# 数据标准化函数


def robust_normalize_data(data):
    # 将数据转换为NumPy数组
    data = np.array(data)
    num_samples, num_timesteps, num_features = data.shape
    data_normalized = np.empty_like(data)

    # 将percentiles字典转换为NumPy数组
    q1_vals = np.array([percentiles[k]['25th'] for k in range(num_features)])
    q2_vals = np.array([percentiles[k]['50th'] for k in range(num_features)])
    q3_vals = np.array([percentiles[k]['75th'] for k in range(num_features)])

    # 使用NumPy的广播特性进行归一化
    data_normalized = (data - q2_vals) / (q3_vals - q1_vals)
    return data_normalized

# 自注意力层
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output
# 定义 CNN+RNN+Attention 模型
class CRNNWithAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CRNNWithAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bilstm = nn.LSTM(input_size=128 + input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                              dropout=0.5, bidirectional=True)

        self.attention = SelfAttention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        original_x = x.clone()  # 保留原始输入
        x = x.permute(0, 2, 1)  # 调整输入形状以适应 CNN
        # 第一层 CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # 第二层 CNN
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.permute(0, 2, 1)  # 调整形状以适应 LSTM
        # 对原始输入进行下采样，使其与卷积后的特征在时间维度上对齐
        original_x = nn.functional.interpolate(original_x.permute(0, 2, 1), size=x.size(1)).permute(0, 2, 1)
        # 连接原始输入和经过卷积的输出
        x = torch.cat((original_x, x), dim=2)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        bilstm_out, _ = self.bilstm(x, (h0, c0))

        attention_out = self.attention(bilstm_out)
        out = self.fc(attention_out[:, -1, :])
        return out

time1 = time.time()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数配置
input_size = 8  # 每个时间步的特征数
hidden_size = 512
num_layers = 3
num_classes = 4  # 标签类别数

# 加载保存的模型
model = CRNNWithAttentionModel(input_size, hidden_size, num_layers, num_classes).to(device)
model_path = '../bestmodel/data-20/minDTW-60-8-robust_normalize_data-x-q2/residual_295.pt'  # 替换为你的模型路径
model.load_state_dict(torch.load(model_path))
model.eval()

time2 = time.time()

loadcost = time2-time1
print(f'loadcost:{loadcost:.2f}')
# 预处理数据
base_dir = '../data/csv_data/finnal-data'  # 替换为你的CSV文件所在目录

data, file_paths, real_labels = load_data_and_labels(base_dir)

time3 = time.time()
# 标签编码
label_encoder = LabelEncoder()
real_labels_encoded = label_encoder.fit_transform(real_labels)

# 标准化数据
standardized_data = robust_normalize_data(data)

# 转换为Tensor
data = torch.tensor(standardized_data, dtype=torch.float32).to(device)

# 创建数据加载器
dataset = TensorDataset(data)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

data_loader_cost = time3 - time2

print(f'data_loader_cost:{data_loader_cost:.2f}')

# 进行分类并输出标签
predicted_labels = []
with torch.no_grad():
    for i, (X_batch,) in enumerate(data_loader):
        outputs = model(X_batch)
        max_index = torch.argmax(outputs).item()
        predicted_labels.append(max_index)  # Extend the list with the batch predictions


file_to_labels = {}
for file_path, predicted_label, real_label in zip(file_paths, predicted_labels, real_labels_encoded):
    if file_path not in file_to_labels:
        file_to_labels[file_path] = {"predicted": [], "real": []}
    file_to_labels[file_path]["predicted"].append(predicted_label)
    file_to_labels[file_path]["real"].append(real_label)

predict_cost = time.time() - time3

print(f'predict_cost:{predict_cost:.2f}')

# 统计不匹配标签的数量
mismatch_counts = Counter()

print("\n预测标签与真实标签不匹配的文件:")
for file_path, labels in list(file_to_labels.items()):  # 使用 list() 以允许修改字典
    mismatches = sum(p != r for p, r in zip(labels["predicted"], labels["real"]))
    if mismatches > 0:
        mismatch_counts[mismatches] += 1
        print(f"{os.path.basename(file_path)} 预测标签: {' '.join(map(str, labels['predicted']))} 真实标签: {' '.join(map(str, labels['real']))}")

# 输出不匹配标签的数目
sum = 0
print("\n不匹配标签的文件数目:")
for count, num_files in sorted(mismatch_counts.items()):
    sum = sum+num_files
    print(f"{count} 个标签不匹配: {num_files} 个文件")
print(f"总计{sum}个文件不匹配")

# 统计各个标签的预测准确率
correct_predictions = {label: 0 for label in label_encoder.classes_}
total_predictions = {label: 0 for label in label_encoder.classes_}

for predicted_label, real_label in zip(predicted_labels, real_labels_encoded):
    total_predictions[label_encoder.inverse_transform([real_label])[0]] += 1
    if predicted_label == real_label:
        correct_predictions[label_encoder.inverse_transform([real_label])[0]] += 1

print("\n每个标签的预测准确率:")
for label in label_encoder.classes_:
    accuracy = correct_predictions[label] / total_predictions[label] * 100
    print(f"{label}: {accuracy:.2f}%")

