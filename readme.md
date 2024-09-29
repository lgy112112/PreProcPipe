
# PreProcPipe

## 项目简介

PreProcPipe 是一个用于医学图像预处理的管道，主要功能包括对医学图像进行加载、裁剪、归一化、重采样以及调整目标尺寸。该项目使用多进程并行处理，支持对 `.nii` 格式的医学图像数据进行处理，并将处理后的结果以 `.npz` 格式保存。

## 主要功能

- 读取医学图像数据（支持 `.nii` 格式）
- 对图像进行非零区域裁剪
- 图像归一化处理（支持 z-score 和 min-max）
- 根据目标体素大小重采样
- 调整图像和分割数据到目标尺寸
- 多进程并行处理加快处理速度

## 文件结构

- `pipeit.py`：主处理脚本，读取 `metadata.csv` 文件中的数据路径，并使用多进程对病例进行处理。
- `pipeline.py`：定义了 `SimplePreprocessor` 类，提供图像的加载、预处理和保存功能。

## 环境要求

- Python 3.6+
- 依赖库：
  - `numpy`
  - `pandas`
  - `nibabel`
  - `tqdm`
  - `scipy`
  - `multiprocessing`
  - `IPython`（用于清理输出）

您可以使用以下命令安装所需依赖：
```bash
pip install numpy pandas nibabel tqdm scipy ipython
```

## 使用方法

### 1. 配置 `metadata.csv`

首先，您需要准备一个 `metadata.csv` 文件，其中包含图像文件的路径和分割文件的路径。`metadata.csv` 文件的格式如下：

| case_id  | image_path                  | label_path                  |
|----------|-----------------------------|-----------------------------|
| case_001 | /path/to/image1.nii.gz       | /path/to/label1.nii.gz       |
| case_002 | /path/to/image2.nii.gz       | /path/to/label2.nii.gz       |

- `case_id`：病例的唯一标识符。
- `image_path`：图像文件的路径（.nii 格式）。
- `label_path`：分割标签文件的路径（如果有）。

### 2. 运行预处理

在项目目录下运行以下命令，启动预处理流程：

```bash
python pipeit.py
```

程序会自动读取 `metadata.csv` 中的文件路径，并依次对每个病例进行预处理。

### 3. 处理进度

在预处理过程中，程序会显示当前处理的病例编号和总病例数。例如：

```
目前正在处理第 1/100 个病例：case_001
```

使用多进程加快处理速度，处理完成后，处理后的图像数据将被保存为 `.npz` 文件，并存储在 `processed_data` 目录下。

### 4. 处理结果

每个处理完的病例，程序会生成以下两个文件并保存到 `processed_data` 目录中：

- `{case_id}_data.npz`：处理后的图像数据。
- `{case_id}_seg.npz`：处理后的分割标签数据（如果有）。

## 示例代码解释

### pipeit.py

该脚本负责读取 `metadata.csv` 中的病例信息，并利用多进程处理每个病例的数据。

```python
# 使用 pandas 读取 CSV 文件
metadata_df = pd.read_csv('metadata.csv')

# 遍历每一行病例数据，并使用多进程进行处理
for idx, row in metadata_df.iterrows():
    process_case((idx, row))
```

`process_case()` 函数主要完成以下任务：
- 加载图像和分割数据
- 创建预处理器实例
- 对数据进行裁剪、归一化、重采样和调整尺寸
- 保存处理后的数据

### pipeline.py

该文件包含了核心的预处理类 `SimplePreprocessor`，它负责对图像和分割数据进行预处理。

- `read_images()`：加载 `.nii` 格式的图像文件。
- `run_case()`：执行预处理操作，包括裁剪、归一化、重采样和调整尺寸。
- `resample_data()`：根据目标体素大小对图像数据进行重采样。
- `resize_to_target_size()`：调整图像或分割数据到目标尺寸。

## 常见问题

### 1. `metadata.csv` 文件格式错误

请确保 `metadata.csv` 中的 `case_id`、`image_path` 和 `label_path` 列已经正确配置，路径应该是文件的完整路径。

### 2. 处理过程中遇到内存不足的问题

如果处理的数据量较大，建议适当减少并行处理的进程数，可以通过修改 `pipeit.py` 中 `mp.Pool(processes=num_processes)` 的 `num_processes` 参数来控制并行进程的数量。

### 3. 处理完成后找不到 `.npz` 文件

确保脚本在执行时已经创建了 `processed_data` 目录，并且 `.npz` 文件的保存路径正确。

## 贡献与反馈

如果您在使用过程中遇到问题或有任何建议，欢迎在 GitHub 项目的 issue 页面提交问题或提供反馈。

## 许可证

本项目遵循 MIT 开源许可证，详细内容请参见 LICENSE 文件。