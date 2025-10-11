import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = '../data/csv_data/combine_data'

def load_data_and_labels(base_dir):
    data = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, skiprows=1)  # Skip header row
                df = df.dropna()  # Drop rows with missing values
                if len(df) > 450:
                    df = df.head(len(df) - 20)  # Discard the last part of the data
                    case_data = df.values  # Directly use the entire data
                    if 0 not in case_data:
                        data.append(case_data)
    return data

def calculate_percentiles(data):
    # 将所有矩阵拼接成一个大的矩阵
    all_data = np.vstack(data)
    percentiles = {}
    detailed_percentiles = {}
    for i in range(all_data.shape[1]):
        column_data = all_data[:, i]
        percentiles[i] = {
            '1st': np.percentile(column_data, 1),
            '5th': np.percentile(column_data, 5),
            '25th': np.percentile(column_data, 25),
            '50th': np.percentile(column_data, 50),
            '75th': np.percentile(column_data, 75),
            '95th': np.percentile(column_data, 95),
            '99th': np.percentile(column_data, 99),
        }
        detailed_percentiles[i] = np.percentile(column_data, np.arange(1, 101))

    return percentiles, detailed_percentiles

data = load_data_and_labels(base_dir)

# 计算百分位数
percentiles, detailed_percentiles = calculate_percentiles(data)

# 打印结果
print("percentiles = {")
for column, values in percentiles.items():
    print(f"    {column}: {{'1st': {values['1st']:.1f}, '5th': {values['5th']:.1f},'25th': {values['25th']:.1f}, '50th': {values['50th']:.1f}, '75th': {values['75th']:.1f} ,'95th': {values['95th']:.1f}, '99th': {values['99th']:.1f},  }},")
print("}")

