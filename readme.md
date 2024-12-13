# PreProcPipe 用于CT/MRI的、适用于多模态影像的预处理流程方法

![xmp](before.png)
![xmp](after.png)

## Choose Language / 选择语言

- [English](readme_en.md)
- [简体中文](readme.md)

## 1. PreProcPipe 项目结构说明

该项目以 BraTS2021 数据集的预处理为例子，主要文件和目录结构如下：

### 数据目录
`PreProcPipe/BraTS2021_Training_Data`  
存放 BraTS2021 数据集的原始训练数据，每个样本以其 ID 创建独立文件夹。

#### 样本目录
- `PreProcPipe/BraTS2021_Training_Data/BraTS2021_00000/`  
  - `BraTS2021_00000_flair/`  
    - 存放该样本的 FLAIR 模态文件，例如：`00000057_brain_flair.nii`  
  - `BraTS2021_00000_seg/`  
    - 存放该样本的分割文件。  
  - `BraTS2021_00000_t1/`  
    - 存放该样本的 T1 模态文件。  
  - `BraTS2021_00000_t1ce/`  
    - 存放该样本的 T1CE 模态文件。  
  - `BraTS2021_00000_t2/`  
    - 存放该样本的 T2 模态文件。  

#### 其他样本
类似结构适用于其他样本，例如：
- `PreProcPipe/BraTS2021_Training_Data/BraTS2021_00002/`  
- `PreProcPipe/BraTS2021_Training_Data/BraTS2021_00003/`  

### 代码和教程
- `PreProcPipe/tutorial.ipynb`  
  - 包含使用 `pipeline.py` 处理数据的详细教程，演示如何加载、预处理和保存结果。刚开始使用的话推荐先看这里的教程。
  
- `PreProcPipe/pipeline.py`  
  - 主预处理脚本，包含用于裁剪、归一化和重采样 BraTS2021 数据的代码逻辑。
  
---

## 2. 代码高光

`SimplePreprocessor` 是一个用于多模态影像预处理的核心类，能够处理多模态 MRI 或 CT 数据，执行裁剪、归一化、重采样和尺寸调整等预处理步骤。

### 预处理步骤和方法

#### 1. 初始化
通过 `__init__` 方法配置预处理参数：
- **`target_spacing`**: 指定目标体素大小（默认 `[1.0, 1.0, 1.0]`）。
- **`normalization_scheme`**: 指定归一化方法（支持 `z-score` 或 `min-max`）。
- **`target_size`**: 指定目标尺寸（如 `[256, 256]`），默认为 `None`（不调整尺寸）。

#### 2. 加载数据
- **`read_images(image_paths)`**: 加载多模态影像数据，返回 NumPy 数组列表和体素间距。
- **`read_seg(seg_path)`**: 加载分割数据，返回 NumPy 数组。

#### 3. 裁剪功能
- **`crop(data_list, seg)`**:
  - 仅裁剪 Z 轴方向的全零区域。
  - 返回裁剪后的影像数据、分割数据以及裁剪属性（裁剪范围和形状变化）。

#### 4. 归一化
- **`_normalize_single_modality(data)`**:
  - 对单个模态数据进行归一化。
  - 支持 `z-score` 和 `min-max` 两种归一化方法。

#### 5. 重采样
- **`compute_new_shape(old_shape, old_spacing, new_spacing)`**:
  - 根据原始形状和体素间距计算目标形状。
  - 输出重采样因子和新形状。
- **`resample_data(data, new_shape, order=3)`**:
  - 重采样影像数据到目标形状。
  - 默认使用三次插值。

#### 6. 调整尺寸
- **`resize_to_target_size(data, target_size, order=3)`**:
  - 将影像数据调整到指定的目标尺寸（如 `[256, 256]`）。
  - 默认保持 Z 轴深度不变。

#### 999. 运行单个样本的预处理-这是上述所有函数的集合
- **`run_case(image_paths, seg_path=None)`**:
  依次执行以下步骤：
  1. **加载数据**：加载所有模态影像和对应的分割数据。
  2. **裁剪 Z 轴**：调用 `crop` 方法，仅裁剪 Z 轴的全零区域，保持其他维度。
  3. **归一化**：对每个模态数据独立归一化（`_normalize_single_modality` 方法）。
  4. **重采样**：使用 `compute_new_shape` 和 `resample_data` 调整体素分辨率。
  5. **调整尺寸**：根据目标尺寸调整数据大小（`resize_to_target_size` 方法）。
  6. **返回结果**：输出裁剪后的数据、分割数据、原始分辨率信息和裁剪属性。

### 处理流程概述
`SimplePreprocessor` 的设计适用于处理具有多模态影像和分割数据的医学影像数据。每个步骤的功能模块化，便于扩展和复用，并支持处理高维数据的大部分常见预处理需求。

---

## 3. 怎么用呢？

`SimplePreprocessor` 提供了一个灵活的接口，适用于多模态和单模态影像数据的预处理流程。以下是具体的使用说明：


### **输入数据格式**

#### **多模态数据**
对于多模态数据（例如 FLAIR、T1、T1CE、T2），输入数据应该是一个文件路径列表，分别指向每个模态的 `.nii` 文件。例如：

```python
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii",
    "BraTS2021_00000/BraTS2021_00000_t1/00000057_brain_t1.nii",
    "BraTS2021_00000/BraTS2021_00000_t1ce/00000057_brain_t1ce.nii",
    "BraTS2021_00000/BraTS2021_00000_t2/00000057_brain_t2.nii"
]
```

#### **单模态数据**
对于单模态数据，输入数据只需包含一个 `.nii` 文件的路径，例如：

```python
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii"
]
```

#### **分割数据**
分割数据的输入是一个单独的文件路径，指向 `.nii` 格式的分割文件。例如：

```python
seg_path = "BraTS2021_00000/BraTS2021_00000_seg/00000057_seg.nii"
```

分割数据是可选的。如果没有分割数据，则将 `seg_path` 设置为 `None`。

---

### **如何调用预处理方法**

#### 1. **初始化预处理器**
先创建一个 `SimplePreprocessor` 实例，可以指定以下参数：
- `target_spacing`：目标体素大小（默认为 `[1.0, 1.0, 1.0]`）。
- `normalization_scheme`：归一化方法（默认为 `"z-score"`）。
- `target_size`：目标尺寸（默认为 `None`，即不调整尺寸）。

例如：

```python
from pipeline import SimplePreprocessor

preprocessor = SimplePreprocessor(
    target_spacing=[1.0, 1.0, 1.0], 
    normalization_scheme="z-score", 
    target_size=[256, 256]
)
```

---

#### 2. **运行单个样本**
使用 `run_case` 方法对单个样本进行预处理：

```python
# 输入数据
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii",
    "BraTS2021_00000/BraTS2021_00000_t1/00000057_brain_t1.nii",
    "BraTS2021_00000/BraTS2021_00000_t1ce/00000057_brain_t1ce.nii",
    "BraTS2021_00000/BraTS2021_00000_t2/00000057_brain_t2.nii"
]
seg_path = "BraTS2021_00000/BraTS2021_00000_seg/00000057_seg.nii"

# 运行预处理
data_list, seg, spacing, properties = preprocessor.run_case(image_paths, seg_path)
```

- `data_list`：预处理后的多模态图像数据（裁剪、归一化、重采样、调整尺寸后）。
- `seg`：预处理后的分割数据（如果有）。
- `spacing`：原始影像的体素间距信息。
- `properties`：裁剪和预处理相关的属性（例如裁剪前后形状、裁剪边界）。

---

#### 3. **运行单模态样本**
对于单模态数据，输入列表只包含一个文件路径：

```python
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii"
]
seg_path = None  # 如果没有分割数据

data_list, seg, spacing, properties = preprocessor.run_case(image_paths, seg_path)
```

此时：
- `data_list` 只包含单模态的处理结果。
- `seg` 为 `None`。

---

### **批量处理样本**
如果需要处理多个样本，可以将每个样本的输入（`image_paths` 和 `seg_path`）存储在列表中，并使用多进程处理工具（例如 `run_in_parallel` 方法）。

```python
from pipeline import run_in_parallel

cases = [
    {
        "image_paths": [
            "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii",
            "BraTS2021_00000/BraTS2021_00000_t1/00000057_brain_t1.nii"
        ],
        "seg_path": "BraTS2021_00000/BraTS2021_00000_seg/00000057_seg.nii"
    },
    {
        "image_paths": [
            "BraTS2021_00001/BraTS2021_00001_flair/00000058_brain_flair.nii"
        ],
        "seg_path": None
    }
]

# 批量处理
results = run_in_parallel(preprocessor, cases, num_workers=4, output_root="preprocessed_data")
```

`results` 返回每个样本的预处理结果。并且，数据会根据你设定的`output_root="preprocessed_data"`存放到相应的文件夹


### **输出数据**
处理完成后，每个样本的返回值包含：
1. **`data_list`**：一个列表，存储处理后的每个模态数据。
2. **`seg`**：处理后的分割数据（如果有）。
3. **`spacing`**：原始的体素间距。
4. **`properties`**：记录裁剪、归一化和重采样的信息，例如：
   ```python
   {
       "shape_before_cropping": [(240, 240, 155), ...],
       "shape_after_cropping": [(240, 240, 120), ...],
       "z_bbox": [10, 130]
   }
   ```

通过这种结构化的返回值，可以轻松地对结果进行后续的保存或分析。

---