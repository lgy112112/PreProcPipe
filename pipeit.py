import pandas as pd
import numpy as np
import pipeline as ppl
import os
from tqdm import tqdm
import multiprocessing as mp
from IPython.display import clear_output  # 用于清除输出

def process_case(args):
    idx, row, total_cases = args
    case_id = row['case_id']
    image_path = row['image_path']
    label_path = row['label_path']  # 如果有分割数据

    # 创建预处理器实例（在每个进程中创建，避免跨进程共享对象的问题）
    preprocessor = ppl.SimplePreprocessor(
        target_spacing=[1.0, 1.0, 1.0],
        normalization_scheme="z-score",
        target_size=[256, 256]  # 根据需要调整
    )

    # 创建保存处理后数据的目录（在主进程中创建过的话，这里可以省略）
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)

    # 清除上一个病例的输出，并打印当前处理进度
    clear_output(wait=True)
    print(f"目前正在处理第 {idx + 1}/{total_cases} 个病例：{case_id}")

    try:
        # 使用预处理器处理当前病例
        data, seg, spacing, properties = preprocessor.run_case(image_path, label_path)

        # 准备保存文件的路径
        data_filename = os.path.join(output_dir, f"{case_id}_data.npz")
        seg_filename = os.path.join(output_dir, f"{case_id}_seg.npz")

        # 将 data 和 seg 保存为 .npz 文件
        np.savez_compressed(data_filename, data=data)
        if seg is not None:
            np.savez_compressed(seg_filename, seg=seg)
        else:
            print(f"病例 {case_id} 没有分割数据。")

        print(f"完成处理病例 {case_id}\n")

    except Exception as e:
        print(f"处理病例 {case_id} 时出错：{e}\n")

if __name__ == '__main__':
    import sys

    # 读取 metadata.csv 文件
    metadata_df = pd.read_csv('metadata.csv')

    # 检查是否有病例需要处理
    if metadata_df.empty:
        print("metadata.csv 文件为空或没有可处理的病例。")
        sys.exit(0)

    # 创建保存处理后数据的目录
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)

    # 获取总的病例数
    total_cases = len(metadata_df)

    # 准备多进程参数列表
    args_list = []
    for idx, row in metadata_df.iterrows():
        args_list.append((idx, row, total_cases))

    # 使用多进程池处理
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示进度条
        for _ in tqdm(pool.imap_unordered(process_case, args_list), total=len(args_list), desc="Processing cases"):
            pass
