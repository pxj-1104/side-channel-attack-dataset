import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import torch.nn.functional as F

path = '../data/dataset/test-4000'
# 创建输出目录
output_dir = '../results/test-4000'
best_model_dir = '../bestmodel/test'
model_name = 'CALSTM-2'

# 设置参数
input_size = 11  # 每个时间步的特征数
hidden_size = 128
num_layers = 2
num_classes = 2  # 标签类别数
num_epochs = 200
batch_size = 256
learning_rate = 0.0001


# 加载数据
def load_data():
    X_train = np.load(path + '/X_train.npy', allow_pickle=True)
    y_train = np.load(path + '/y_train.npy', allow_pickle=True)
    X_test = np.load(path + '/X_test.npy', allow_pickle=True)
    y_test = np.load(path + '/y_test.npy', allow_pickle=True)
    high_test_data = np.load(path + '/high_test_data.npy', allow_pickle=True)
    high_test_labels = np.load(path + '/high_test_labels.npy', allow_pickle=True)
    medium_test_data = np.load(path + '/medium_test_data.npy', allow_pickle=True)
    medium_test_labels = np.load(path + '/medium_test_labels.npy', allow_pickle=True)
    low_test_data = np.load(path + '/low_test_data.npy', allow_pickle=True)
    low_test_labels = np.load(path + '/low_test_labels.npy', allow_pickle=True)
    return X_train, y_train, X_test, y_test, high_test_data, high_test_labels, medium_test_data, medium_test_labels, low_test_data, low_test_labels


# 数据预处理
def preprocess_data(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = torch.tensor(y, dtype=torch.long)
    return X, y, le

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DualAttention(nn.Module):
    def __init__(self, input_dim):
        super(DualAttention, self).__init__()
        self.input_dim = input_dim

        # 时间维度的自注意力
        self.W_q_time = nn.Linear(input_dim, input_dim)
        self.W_k_time = nn.Linear(input_dim, input_dim)
        self.W_v_time = nn.Linear(input_dim, input_dim)

        # 特征维度的自注意力
        self.W_q_feat = nn.Linear(input_dim, input_dim)
        self.W_k_feat = nn.Linear(input_dim, input_dim)
        self.W_v_feat = nn.Linear(input_dim, input_dim)

        self.scale = torch.sqrt(torch.FloatTensor([input_dim])).to(device)

    def forward(self, x):
        # 输入: x (batch_size, time_steps, input_dim)

        # 时间维度注意力
        Q_time = self.W_q_time(x)  # (batch_size, time_steps, input_dim)
        K_time = self.W_k_time(x)  # (batch_size, time_steps, input_dim)
        V_time = self.W_v_time(x)  # (batch_size, time_steps, input_dim)

        attention_scores_time = torch.matmul(Q_time, K_time.transpose(-2, -1)) / self.scale
        attention_weights_time = torch.nn.functional.softmax(attention_scores_time, dim=-1)
        attention_output_time = torch.matmul(attention_weights_time, V_time)  # (batch_size, time_steps, input_dim)

        # 特征维度注意力
        x_transpose = x.transpose(1, 2)  # 转置为 (batch_size, input_dim, time_steps)
        Q_feat = self.W_q_feat(x_transpose.permute(0, 2, 1))  # (batch_size, time_steps, input_dim)
        K_feat = self.W_k_feat(x_transpose.permute(0, 2, 1))  # (batch_size, time_steps, input_dim)
        V_feat = self.W_v_feat(x_transpose.permute(0, 2, 1))  # (batch_size, time_steps, input_dim)

        attention_scores_feat = torch.matmul(Q_feat, K_feat.transpose(-2, -1)) / self.scale
        attention_weights_feat = torch.nn.functional.softmax(attention_scores_feat, dim=-1)
        attention_output_feat = torch.matmul(attention_weights_feat, V_feat)  # (batch_size, time_steps, input_dim)

        # 融合时间和特征维度注意力结果
        attention_output = torch.cat((attention_output_time, attention_output_feat), dim=2)  # (batch_size, time_steps, input_dim * 2)

        return attention_output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# CRNNWithAttentionModel 模型
class CRNNWithAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CRNNWithAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        # ResNet blocks
        self.block1 = ResidualBlock(input_size, 64)
        self.block2 = ResidualBlock(64 + input_size, 128)
        self.block3 = ResidualBlock(128 + input_size, 256)

        # Attention
        self.attention_input_dim = 256 + input_size
        self.attention = DualAttention(input_dim=self.attention_input_dim)

        # BiLSTM 输入维度是 attention 输出 + 原始输入
        self.bilstm_input_dim = self.attention_input_dim * 2 + input_size
        self.bilstm = nn.LSTM(
            input_size=self.bilstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        original_x = x.clone()  # [B, T, 11]
        x = x.permute(0, 2, 1)  # [B, 11, T]

        # Block 1
        x = self.block1(x)
        res1 = F.interpolate(original_x.permute(0, 2, 1), size=x.shape[2])
        x = torch.cat((x, res1), dim=1)  # [B, 64+11, T]

        # Block 2
        x = self.block2(x)
        res2 = F.interpolate(original_x.permute(0, 2, 1), size=x.shape[2])
        x = torch.cat((x, res2), dim=1)  # [B, 128+11, T]

        # Block 3
        x = self.block3(x)
        res3 = F.interpolate(original_x.permute(0, 2, 1), size=x.shape[2])
        x = torch.cat((x, res3), dim=1)  # [B, 256+11, T]

        # 准备 Attention 输入
        x = x.permute(0, 2, 1)  # [B, T, 256+11]
        attention_out = self.attention(x)  # [B, T, (256+11)*2]

        # 拼接原始输入
        attention_plus_orig = torch.cat((attention_out, original_x), dim=2)  # [B, T, (256+11)*2 + 11]

        # BiLSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.bilstm(attention_plus_orig, (h0, c0))

        out = self.fc(lstm_out[:, -1, :])
        return out

# 分别测试高、中、低负载数据集
def evaluate_on_specific_load(model, data_loader):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        false_positive_rate = cm.sum(axis=0) - np.diag(cm)
        false_negative_rate = cm.sum(axis=1) - np.diag(cm)
        false_positive_rate = false_positive_rate / cm.sum(axis=1).sum()
        false_negative_rate = false_negative_rate / cm.sum(axis=0).sum()
        return accuracy, false_positive_rate.mean(), false_negative_rate.mean(), cm


# 训练和测试函数
def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, num_epochs):
    maxacc = 0
    best_epoch = 0
    best_model_state = None
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    with open(f'{output_dir}/{model_name}.txt', 'w') as f:
        for epoch in range(num_epochs):
            model.train()
            start_time = time.time()
            for i, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.extend(y_batch.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

                cm = confusion_matrix(y_true, y_pred)

                fp = cm.sum(axis=0) - np.diag(cm)  # False Positives
                fn = cm.sum(axis=1) - np.diag(cm)  # False Negatives
                tp = np.diag(cm)  # True Positives
                tn = cm.sum() - (fp + fn + tp)  # True Negatives
                accuracy = 100 * np.trace(cm) / np.sum(cm)
                high_acc, high_fp_rate, high_fn_rate, high_cm = evaluate_on_specific_load(model, high_test_loader)
                medium_acc, medium_fp_rate, medium_fn_rate, medium_cm = evaluate_on_specific_load(model, medium_test_loader)
                low_acc, low_fp_rate, low_fn_rate, low_cm = evaluate_on_specific_load(model, low_test_loader)

                # 计算训练集上的准确率
                train_accuracy, _, _, _ = evaluate_on_specific_load(model, train_loader)
                end_time = time.time()
                time_cost = end_time - start_time

                if accuracy > maxacc:
                    maxacc = accuracy
                    best_model_state = model.state_dict()
                    if maxacc > 85:
                        torch.save(best_model_state, f'{best_model_dir}/{model_name}_{str(epoch)}.pt')
                        best_epoch = epoch

                output_1 = (
                    f'Epoch [{epoch + 1}/{num_epochs}], time: {time_cost:.2f}, Loss: {loss.item():.4f}, Train_accuracy:{train_accuracy * 100:.2f}%, Test Accuracy: {accuracy:.2f}% '
                    f'High: {high_acc:.4f}, Medium: {medium_acc:.4f}, Low: {low_acc:.4f}, Max Accuracy: {maxacc:.2f}%'
                )
                output_2 = (
                    f'Epoch [{epoch + 1}/{num_epochs}], time: {time_cost:.2f},Loss: {loss.item():.4f}, Train_accuracy:{train_accuracy * 100:.2f}%, Test Accuracy: {accuracy:.2f}% '
                    f'High: {high_acc:.4f}, Medium: {medium_acc:.4f}, Low: {low_acc:.4f}, Max Accuracy: {maxacc:.2f}%\n'
                    f'All Confusion Matrix:\n{cm}\n'
                    f'high Confusion Matrix:\n{high_cm}\n'
                    f'medium Confusion Matrix:\n{medium_cm}\n'
                    f'low Confusion Matrix:\n{low_cm}\n\n'
                )
                print(output_1)
                f.write(output_2)

    # 保存最佳模型
    torch.save(best_model_state, f'{best_model_dir}/{model_name}_{str(best_epoch)}.pt')

# 加载和预处理数据
X_train, y_train, le = preprocess_data(*load_data()[:2])
X_test, y_test = preprocess_data(*load_data()[2:4])[:2]
high_test_data, high_test_labels = preprocess_data(*load_data()[4:6])[:2]
medium_test_data, medium_test_labels = preprocess_data(*load_data()[6:8])[:2]
low_test_data, low_test_labels = preprocess_data(*load_data()[8:])[:2]

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
high_test_dataset = TensorDataset(high_test_data, high_test_labels)
medium_test_dataset = TensorDataset(medium_test_data, medium_test_labels)
low_test_dataset = TensorDataset(low_test_data, low_test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
high_test_loader = DataLoader(high_test_dataset, batch_size=batch_size, shuffle=False)
medium_test_loader = DataLoader(medium_test_dataset, batch_size=batch_size, shuffle=False)
low_test_loader = DataLoader(low_test_dataset, batch_size=batch_size, shuffle=False)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = CRNNWithAttentionModel(input_size, hidden_size, num_layers, num_classes).to(device)
print(f"模型参数总数: {count_parameters(model):,}")

class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和测试模型
train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, num_epochs)

