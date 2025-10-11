import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,RobustScaler

base_dir = '../data/csv_data/testdata'
save_dir = '../data/dataset/group-data/separate-4-120'


percentiles = { #cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-icache-load-misses,LLC-loads,LLC-load-misses,LLC-store-misses
    0: {'1st': 250965.0, '5th': 363882.0,'25th': 902376.0, '50th': 2957030.5 ,'75th': 6660350.0 ,'95th': 14434327.0, '99th': 22337618.9,  },
    1: {'1st': 923.0, '5th': 1486.0,'25th': 4924.0, '50th': 130003.0 ,'75th': 606428.0 ,'95th': 3592487.1, '99th': 6176059.6,  },
    2: {'1st': 487484.0, '5th': 1824049.9,'25th': 20210061.5, '50th': 37898381.5, '75th': 61534066.8 ,'95th': 107391350.0, '99th': 143820937.4,  },
    3: {'1st': 37056.0, '5th': 55110.0,'25th': 320507.0, '50th': 1304914.0, '75th': 2985272.5 ,'95th': 6459636.2, '99th': 22905894.5,  },
    4: {'1st': 116896.0, '5th': 177093.0,'25th': 340360.0, '50th': 1277704.5, '75th': 3783308.2 ,'95th': 8714825.2, '99th': 12625176.2,  },
    5: {'1st': 13727.0, '5th': 21781.0,'25th': 89136.0, '50th': 254444.0, '75th': 646203.0 ,'95th': 1617420.0, '99th': 3697958.4,  },
    6: {'1st': 162.0, '5th': 320.0,'25th': 806.0, '50th': 8044.0, '75th': 68963.0 ,'95th': 349035.0, '99th': 1022372.0,  },
    7: {'1st': 219.0, '5th': 514.0,'25th': 1020.0, '50th': 6513.0, '75th': 81318.0 ,'95th': 770937.1, '99th': 2505278.9,  },
}


def get_label_from_path(file_path):
    parts = file_path.split(os.sep)
    if 'attack' in parts:
        return parts[parts.index('attack') + 1][4:] #将同类攻击归为一类
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


def load_data_and_labels(base_dir):
    count = 0
    data = []
    labels = []
    load_types = []
    file_identifiers = []  # Keep track of which file each segment comes from

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, skiprows=1)  # Skip header row
                df = df.dropna()  # Drop rows with missing values
                if len(df) > 500:
                    if 'aes_flush+reload' in file_path:
                        df = df.head(480)
                    else:
                        df = df.head(len(df) - 10)  # Discard the last 10 rows

                    label = get_label_from_path(file_path)
                    load_type = get_load_type_from_path(file_path)

                    num_cases = ((len(df)) // 60) - 1  # Sliding window logic
                    for i in range(num_cases):
                        start_row = i * 60
                        end_row = start_row + 120
                        case_data = df.iloc[start_row:end_row].values
                        if 0 not in case_data:  # Check for zeros
                            data.append(case_data)
                            labels.append(label)
                            load_types.append(load_type)
                            file_identifiers.append(file)  # Use the file name as the identifier
            count = count +1
            print(f'\rfinish {count} files', end='')


    return data, labels, load_types, file_identifiers

def split_data_by_files(data, labels, load_types, file_identifiers, test_size=0.2):#用于防止数据不泄露
    # Get unique file identifiers
    unique_files = list(set(file_identifiers))

    # Split based on unique file identifiers
    train_files, test_files = train_test_split(
        unique_files, test_size=test_size, stratify=[labels[file_identifiers.index(file)] for file in unique_files])

    X_train, X_test, y_train, y_test, load_train, load_test = [], [], [], [], [], []

    # Assign all segments from the same file to either train or test set
    for i, file in enumerate(file_identifiers):
        if file in train_files:
            X_train.append(data[i])
            y_train.append(labels[i])
            load_train.append(load_types[i])
        else:
            X_test.append(data[i])
            y_test.append(labels[i])
            load_test.append(load_types[i])

    return X_train, X_test, y_train, y_test, load_train, load_test


def encode_labels(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder



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
    data_normalized = (data - q1_vals) / (q3_vals - q1_vals)
    return data_normalized

def standardize_data_empirical_value(data):
    data = np.array(data)
    num_samples, num_timesteps, num_features = data.shape
    data_normalized = np.empty_like(data)

    # 将percentiles字典转换为NumPy数组
    min_vals = np.array([percentiles[k]['1st'] for k in range(num_features)])
    max_vals = np.array([percentiles[k]['95th'] for k in range(num_features)])

    # 使用NumPy的广播特性进行归一化
    data_normalized = (data - min_vals) / (max_vals - min_vals)
    # 由于采用百分经验值进行归一化，可能使得部分数值小于0，这里如果值小于0，我们之间将其截断为0
    data_normalized[data_normalized < 0] = 0
    return data_normalized

def standardize_data_log(data):
    data = np.array(data)
    num_samples, num_timesteps, num_features = data.shape
    data_normalized = np.empty_like(data)

    # 将percentiles字典转换为NumPy数组
    min_vals = np.array([percentiles[k]['1st'] for k in range(num_features)])
    max_vals = np.array([percentiles[k]['95th'] for k in range(num_features)])
    upper_bound = np.array([percentiles[k]['99th'] for k in range(num_features)])

    # 使用NumPy的广播特性进行归一化
    data_normalized = (data - min_vals) / (max_vals - min_vals)

    # 对超过95分位的值进行处理
    for i in range(num_features):
        mask = data[:, :, i] > max_vals[i]
        eps = 1e-6
        data_normalized[:, :, i][mask] = 1 + eps + ( np.log1p(data[:, :, i][mask] - max_vals[i]) / np.log1p(upper_bound[i] - max_vals[i]))

    # 由于采用百分经验值进行归一化，可能使得部分数值小于0，这里如果值小于0，我们之间将其截断为0
    data_normalized[data_normalized < 0] = 0

    return data_normalized

def standardize_data(data):
    scaler = RobustScaler()
    num_samples = len(data)
    num_timesteps = len(data[0])
    num_features = len(data[0][0])

    data_reshaped = np.array(data).reshape(-1, num_features)  # Reshape to (samples*time_steps, features)
    data_standardized = scaler.fit_transform(data_reshaped)
    data_standardized = data_standardized.reshape(num_samples, num_timesteps, num_features)

    return data_standardized, scaler


# Load data and labels

data, labels, load_types, file_identifiers = load_data_and_labels(base_dir)

encoded_labels, label_encoder = encode_labels(labels)
# Standardize data
standardized_data = robust_normalize_data(data)

# Split data
X_train, X_test, y_train, y_test, load_train, load_test = split_data_by_files(
    standardized_data, encoded_labels, load_types, file_identifiers)

# Separate test data by load type
high_test_data = [X_test[i] for i in range(len(X_test)) if load_test[i] == 'high']
medium_test_data = [X_test[i] for i in range(len(X_test)) if load_test[i] == 'medium']
low_test_data = [X_test[i] for i in range(len(X_test)) if load_test[i] == 'low']

high_test_labels = [y_test[i] for i in range(len(y_test)) if load_test[i] == 'high']
medium_test_labels = [y_test[i] for i in range(len(y_test)) if load_test[i] == 'medium']
low_test_labels = [y_test[i] for i in range(len(y_test)) if load_test[i] == 'low']


# Print example
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
print("High load test set size:", len(high_test_data))
print("Medium load test set size:", len(medium_test_data))
print("Low load test set size:", len(low_test_data))
print("data shape ",X_train[0].shape)
# Create save directory (if it doesn't exist)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save datasets as .npy files
np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
np.save(os.path.join(save_dir, 'high_test_data.npy'), high_test_data)
np.save(os.path.join(save_dir, 'high_test_labels.npy'), high_test_labels)
np.save(os.path.join(save_dir, 'medium_test_data.npy'), medium_test_data)
np.save(os.path.join(save_dir, 'medium_test_labels.npy'), medium_test_labels)
np.save(os.path.join(save_dir, 'low_test_data.npy'), low_test_data)
np.save(os.path.join(save_dir, 'low_test_labels.npy'), low_test_labels)


