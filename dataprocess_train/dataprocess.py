import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# 定义文件读取和处理的函数
def load_and_process_data(file_path, save_dir):
    # 获取上一级目录名称，用于命名CSV文件
    parent_dir_name = os.path.basename(os.path.dirname(file_path))

    # 定义需要解析的列名
    columns = ['cache-references', 'cache-misses', 'L1-dcache-loads',
               'L1-dcache-load-misses', "L1-icache-load-misses", 'LLC-loads', 'LLC-load-misses', 'LLC-store-misses']

    # 用来存储解析后的数据
    parsed_data = []

    # 临时字典存储每个时间点的事件
    event_row = {}

    # 当前时间戳，用来区分不同的事件组
    current_time = None

    # 打开文件并逐行读取
    with open(file_path, 'r') as file:
        skipflie = False
        for line in file:

            # 去掉两端的空白符
            line = line.strip()

            # 跳过注释和空行
            if line.startswith("#") or line == "":
                continue

            if 'not count' in line:
                skipflie = True
                break


            # 按空白字符进行分割，数据结构有严格的格式，数据在固定位置
            parts = line.split()

            # 提取时间戳、计数和事件名
            time = parts[0]

            count = float(parts[1].replace(',', ''))  # 将计数转换为浮点数并去掉逗号
            event = parts[2]

            # 如果时间戳变化，说明是一组新的事件，将前一组事件存储起来
            if current_time != time:
                if event_row:
                    # 按照定义的列顺序存储事件数据，如果某个事件没有出现则默认为 0
                    parsed_data.append([event_row.get(col, 0.0) for col in columns])
                # 清空事件字典并更新时间戳
                event_row = {}
                current_time = time

            # 将当前事件的计数值存储在相应的列名中
            if event == 'cache-references':
                event_row['cache-references'] = count
            elif event == 'cache-misses':
                event_row['cache-misses'] = count
            elif event == 'L1-dcache-loads':
                event_row['L1-dcache-loads'] = count
            elif event == 'L1-dcache-load-misses':
                event_row['L1-dcache-load-misses'] = count
            elif event == 'L1-icache-load-misses':
                event_row['L1-icache-load-misses'] = count
            elif event == 'LLC-loads':
                event_row['LLC-loads'] = count
            elif event == 'LLC-load-misses':
                event_row['LLC-load-misses'] = count
            elif event == 'LLC-store-misses':
                event_row['LLC-store-misses'] = count

        # 别忘了处理最后一组事件
        if event_row:
            parsed_data.append([event_row.get(col, 0.0) for col in columns])

    if not skipflie:
        # 将数据转换为 DataFrame
        df = pd.DataFrame(parsed_data, columns=columns)

        # 构建输出目录路径
        relative_path = os.path.relpath(os.path.dirname(file_path), base_dir)  # 相对路径
        output_dir = os.path.join(save_dir, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

        saveid = 0
        # 构建输出CSV的路径，文件名为上一级文件夹的名称
        if int(parent_dir_name)> 518500:
            saveid = int(parent_dir_name) - 518500 + 52000
        else:
            saveid = int(parent_dir_name)-26500
        output_csv_path = os.path.join(output_dir, f'{saveid}.csv')

        # 保存为 CSV 文件
        df.to_csv(output_csv_path, index=False)
        print(f"Data saved to {output_csv_path}")


# 获取所有目录并处理文件
def get_all_files(base_dir):
    all_files = []

    # 遍历所有文件夹，找到hardware_events.txt文件
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'hardware_events.txt':
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    return all_files


def process_files_in_parallel(base_dir, save_dir):
    # 获取所有的文件路径
    all_files = get_all_files(base_dir)

    # 使用 ThreadPoolExecutor 并发处理目录
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有任务
        futures = [executor.submit(load_and_process_data, file, save_dir) for file in all_files]

        # 等待所有任务完成
        for future in futures:
            future.result()  # 获取任务结果，捕获异常


# 示例：处理整个数据集
base_dir = '../data/initial_data/add_data-11-25'
save_dir = '../data/csv_data/add_data-12-18'
process_files_in_parallel(base_dir, save_dir)
