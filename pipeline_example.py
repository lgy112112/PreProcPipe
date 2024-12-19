import pandas as pd
from pipeline import SimplePreprocessor as ppp
from pipeline import run_in_parallel


# 定义读取 metadata.csv 并生成 cases 列表的函数
def load_cases_from_metadata(csv_path):
    """
    从 metadata.csv 加载病例信息，并生成 (image_paths, seg_path) 的列表。
    
    参数：
    - csv_path: metadata.csv 文件路径。
    
    返回：
    - cases: 包含病例信息的列表，每个元素是一个字典，格式为：
      {
          "sample_id": 样本ID,
          "image_paths": [模态1路径, 模态2路径, ...],
          "seg_path": 分割路径或 None
      }
    """
    df = pd.read_csv(csv_path)
    cases = []
    for _, row in df.iterrows():
        # 提取模态路径
        image_paths = [row['t1_path'], row['t1ce_path'], row['t2_path'], row['flair_path']]
        # 过滤掉空值
        image_paths = [path for path in image_paths if pd.notnull(path)]
        # 提取分割路径
        seg_path = row['seg_path'] if pd.notnull(row['seg_path']) else None
        # 添加到 cases
        cases.append({
            "sample_id": row['sample_id'],
            "image_paths": image_paths,
            "seg_path": seg_path
        })
    return cases


if __name__ == "__main__":
    example_preprocessor = ppp(
        target_spacing = [0.5, 0.5, 0.5],
        target_size = [256, 256],
        normalization_scheme = "min-max",
    )

    cases = load_cases_from_metadata(r"D:\REPO\PreProcPipe\BraTS2021_Training_Data\metadata.csv")
    results = run_in_parallel(example_preprocessor, cases, num_workers=8, output_root="preprocessed_data")



