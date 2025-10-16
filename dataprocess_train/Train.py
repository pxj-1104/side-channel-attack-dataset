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

# ==============================
# Path configurations
# ==============================
path = ''
output_dir = ''
best_model_dir = ''
model_name = ''

# Example alternative configuration
# path = '/root/autodl-tmp/dataset/'
# output_dir = '../results/light-CLAM'
# best_model_dir = '../bestmodel/light-CLAM'
# model_name = 'light-CLAM'

# ==============================
# Model and training parameters
# ==============================
input_size = 11
hidden_size = 128
num_layers = 2
num_classes = 2
num_epochs = 1000

batch_size = 256
learning_rate = 0.0001


# ==============================
# Data loading
# ==============================
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


# ==============================
# Data preprocessing
# ==============================
def preprocess_data(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = torch.tensor(y, dtype=torch.long)
    return X, y, le


# ==============================
# Parameter counting utilities
# ==============================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model):
    print("\nParameters per layer:")
    print("-" * 50)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"Layer: {name}, Parameters: {num_params:,}")
            total_params += num_params
    print(f"\nTotal Parameters: {total_params:,}")
    return total_params


# ==============================
# Dual Attention module
# ==============================
class DualAttention(nn.Module):
    def __init__(self, input_dim):
        super(DualAttention, self).__init__()
        self.input_dim = input_dim

        # Temporal attention
        self.W_q_time = nn.Linear(input_dim, input_dim)
        self.W_k_time = nn.Linear(input_dim, input_dim)
        self.W_v_time = nn.Linear(input_dim, input_dim)

        # Feature-wise attention
        self.W_q_feat = nn.Linear(input_dim, input_dim)
        self.W_k_feat = nn.Linear(input_dim, input_dim)
        self.W_v_feat = nn.Linear(input_dim, input_dim)

        self.scale = torch.sqrt(torch.FloatTensor([input_dim])).to(device)

    def forward(self, x):
        # --- Temporal attention ---
        Q_time = self.W_q_time(x)
        K_time = self.W_k_time(x)
        V_time = self.W_v_time(x)

        attention_scores_time = torch.matmul(Q_time, K_time.transpose(-2, -1)) / self.scale
        attention_weights_time = torch.nn.functional.softmax(attention_scores_time, dim=-1)
        attention_output_time = torch.matmul(attention_weights_time, V_time)

        # --- Feature attention ---
        x_transpose = x.transpose(1, 2)
        Q_feat = self.W_q_feat(x_transpose.permute(0, 2, 1))
        K_feat = self.W_k_feat(x_transpose.permute(0, 2, 1))
        V_feat = self.W_v_feat(x_transpose.permute(0, 2, 1))

        attention_scores_feat = torch.matmul(Q_feat, K_feat.transpose(-2, -1)) / self.scale
        attention_weights_feat = torch.nn.functional.softmax(attention_scores_feat, dim=-1)
        attention_output_feat = torch.matmul(attention_weights_feat, V_feat)

        # Concatenate temporal and feature-level attention outputs
        attention_output = torch.cat((attention_output_time, attention_output_feat), dim=2)
        return attention_output


# ==============================
# CNN + LSTM + Dual Attention Model
# ==============================
class CRNNWithAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CRNNWithAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --- Convolutional layers ---
        self.avgpool1 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32 + input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=64 + input_size, out_channels=96, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # --- Dual Attention layer ---
        self.attention = DualAttention(input_dim=96 + input_size)

        # --- LSTM layer ---
        self.lstm = nn.LSTM(input_size=(96 + input_size) * 2, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=0.5, bidirectional=False)

        # --- Fully connected output layer ---
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        original_x = x.clone()
        x = x.permute(0, 2, 1)  # [batch, input_size, seq_len]

        x = self.avgpool1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = torch.cat((x, original_x.permute(0, 2, 1)), dim=1)

        x = self.conv2(x)
        x = self.relu2(x)
        x = torch.cat((x, original_x.permute(0, 2, 1)), dim=1)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.permute(0, 2, 1)
        x = torch.cat((original_x, x), dim=2)
        attention_out = self.attention(x)

        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, attention_out.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, attention_out.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(attention_out, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out


# ==============================
# Evaluation on specific workload levels
# ==============================
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


# ==============================
# Model training and evaluation loop
# ==============================
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

            # --- Evaluation phase ---
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
                accuracy = 100 * np.trace(cm) / np.sum(cm)
                high_acc, _, _, high_cm = evaluate_on_specific_load(model, high_test_loader)
                medium_acc, _, _, medium_cm = evaluate_on_specific_load(model, medium_test_loader)
                low_acc, _, _, low_cm = evaluate_on_specific_load(model, low_test_loader)
                train_accuracy, _, _, _ = evaluate_on_specific_load(model, train_loader)

                end_time = time.time()
                time_cost = end_time - start_time

                # Save the best model
                if accuracy > maxacc:
                    maxacc = accuracy
                    best_model_state = model.state_dict()
                    if maxacc > 85:
                        torch.save(best_model_state, f'{best_model_dir}/{model_name}_{str(epoch)}.pt')
                        best_epoch = epoch

                output_1 = (
                    f'Epoch [{epoch + 1}/{num_epochs}], time: {time_cost:.2f}, Loss: {loss.item():.4f}, '
                    f'Train_accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {accuracy:.2f}% '
                    f'High: {high_acc:.4f}, Medium: {medium_acc:.4f}, Low: {low_acc:.4f}, Max Accuracy: {maxacc:.2f}%'
                )
                output_2 = (
                    f'{output_1}\nAll Confusion Matrix:\n{cm}\n'
                    f'High Confusion Matrix:\n{high_cm}\n'
                    f'Medium Confusion Matrix:\n{medium_cm}\n'
                    f'Low Confusion Matrix:\n{low_cm}\n\n'
                )
                print(output_1)
                f.write(output_2)

    torch.save(best_model_state, f'{best_model_dir}/{model_name}_{str(best_epoch)}.pt')


# ==============================
# Load and preprocess datasets
# ==============================
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

# ==============================
# Device setup
# ==============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================
# Model initialization
# ==============================
model = CRNNWithAttentionModel(input_size, hidden_size, num_layers, num_classes).to(device)

# Print layer-wise parameter count
count_parameters_per_layer(model)

# Print total parameter count
print(f"\nTotal model parameters: {count_parameters(model):,}")

# ==============================
# Loss function and optimizer
# ==============================
class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==============================
# Train and evaluate the model
# ==============================
train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, num_epochs)
