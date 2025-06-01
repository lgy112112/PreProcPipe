# PreProcPipe: 多模态医学影像预处理框架

一个用于CT/MRI等多模态医学影像预处理的高效框架，支持自动化的数据预处理流程和元数据生成。

## Choose Language / 选择语言

- [English](readme_en.md)
- [简体中文](readme.md)

![预处理前](before.png)
![预处理后](after.png)

## 目录

- [主要功能](#主要功能)
- [技术架构](#技术架构)
- [代码实现](#代码实现)
- [使用指南](#使用指南)
- [配置与扩展](#配置与扩展)
- [示例](#示例)

## 主要功能

### 1. 预处理管道 (pipeline.py)
- 多模态医学影像数据处理
- Z轴方向智能裁剪
- 可配置的数据归一化
- 灵活的图像重采样
- 并行处理支持
- 自动保存处理结果

### 2. 元数据自动生成 (LLM_metadata.py)
- 目录结构智能分析
- LLM驱动的元数据规则生成
- 自动验证和错误检测
- DeepSeek API集成

## 技术架构

### 预处理管道架构

预处理管道采用模块化设计，按照以下步骤顺序处理数据：

1. **数据输入** → **SimplePreprocessor**
   - 接收多模态医学影像数据
   - 支持.nii格式文件

2. **数据加载**
   - 读取多模态图像数据
   - 读取分割数据（如果有）

3. **Z轴裁剪**
   - 智能识别有效区域
   - 去除冗余空白区域

4. **归一化**
   - 支持z-score标准化
   - 支持min-max归一化

5. **重采样**
   - 调整体素间距
   - 保持图像质量

6. **尺寸调整**
   - 统一输出尺寸
   - 可选的尺寸配置

7. **输出处理后的数据**
   - 保存为标准格式
   - 生成处理元数据

### 元数据生成系统架构

元数据生成系统采用LLM驱动的智能分析流程：

1. **数据集根目录** → **目录结构分析**
   - 扫描文件系统
   - 识别文件组织模式

2. **随机采样**
   - 选取代表性样本
   - 分析文件命名规律

3. **DeepSeek API调用**
   - 发送结构化请求
   - 接收AI分析结果

4. **生成处理代码**
   - 自动生成Python脚本
   - 包含数据处理逻辑

5. **执行与验证**
   - 运行生成的代码
   - 检查处理结果

6. **生成metadata.csv**
   - 记录数据集信息
   - 建立索引关系

## 代码实现

### 核心类：SimplePreprocessor

```python
class SimplePreprocessor:
    def __init__(self, target_spacing=[1.0, 1.0, 1.0], 
                 normalization_scheme="z-score", 
                 target_size=None):
        """
        初始化预处理器
        
        参数：
        - target_spacing: 目标体素大小，默认[1.0, 1.0, 1.0]
        - normalization_scheme: 归一化方案("z-score"/"min-max")
        - target_size: 目标尺寸，如[256, 256]
        """
```

#### 主要方法：

1. **数据加载**
```python
def read_images(self, image_paths):
    """加载多模态图像数据"""

def read_seg(self, seg_path):
    """加载分割数据"""
```

2. **数据预处理**
```python
def crop(self, data_list, seg):
    """Z轴智能裁剪"""

def _normalize_single_modality(self, data):
    """单模态数据归一化"""

def compute_new_shape(self, old_shape, old_spacing, new_spacing):
    """计算重采样目标形状"""

def resample_data(self, data, new_shape, order=3):
    """数据重采样"""

def resize_to_target_size(self, data, target_size, order=3):
    """调整数据尺寸"""
```

3. **处理流程**
```python
def run_case(self, image_paths, seg_path=None):
    """执行完整的预处理流程"""
```

### 元数据生成系统

```python
def analyze_directory(root_directory, sample_folder_count=5, sample_file_count=10):
    """分析目录结构并采样"""

def generate_metadata(root_directory, your_api_key=None):
    """使用LLM生成元数据处理代码"""

def execute_metadata_script(root_directory):
    """执行并验证元数据生成"""
```

## 使用指南

### 1. 基础预处理流程

```python
from pipeline import SimplePreprocessor

# 初始化预处理器
preprocessor = SimplePreprocessor(
    target_spacing=[1.0, 1.0, 1.0],
    normalization_scheme="z-score",
    target_size=[256, 256]
)

# 准备数据路径
image_paths = [
    "path/to/flair.nii",
    "path/to/t1.nii",
    "path/to/t1ce.nii",
    "path/to/t2.nii"
]
seg_path = "path/to/seg.nii"

# 执行预处理
data_list, seg, spacing, properties = preprocessor.run_case(image_paths, seg_path)
```

### 2. 批量处理

```python
from pipeline import run_in_parallel

# 准备多个样本
cases = [
    {
        "sample_id": "case_001",
        "image_paths": ["path/to/case1/flair.nii", ...],
        "seg_path": "path/to/case1/seg.nii"
    },
    # 更多样本...
]

# 并行处理
results = run_in_parallel(preprocessor, cases, "output_dir", num_workers=4)
```

### 3. 元数据生成

```python
from LLM_metadata import analyze_directory, generate_metadata, execute_metadata_script

# 分析目录结构
analyze_directory(root_directory="dataset_path", 
                 sample_folder_count=5, 
                 sample_file_count=10)

# 生成元数据代码
generate_metadata(root_directory="dataset_path", 
                 your_api_key="your-deepseek-api-key")

# 执行生成的代码
execute_metadata_script(root_directory="dataset_path")
```

## 配置与扩展

### 1. 预处理配置

可以通过修改SimplePreprocessor的初始化参数来自定义预处理行为：

- `target_spacing`: 调整目标体素大小
- `normalization_scheme`: 选择归一化方案
- `target_size`: 设置输出尺寸

### 2. 扩展功能

#### 添加新的归一化方法：

```python
def _normalize_custom(self, data):
    """
    自定义归一化方法
    """
    # 实现你的归一化逻辑
    return normalized_data

# 在SimplePreprocessor中添加
if self.normalization_scheme == "custom":
    data = self._normalize_custom(data)
```

#### 添加新的预处理步骤：

```python
def new_preprocessing_step(self, data):
    """
    新的预处理步骤
    """
    # 实现新的预处理逻辑
    return processed_data

# 在run_case方法中添加
data_list = [self.new_preprocessing_step(d) for d in data_list]
```

## 示例

### 1. 单模态CT图像预处理

```python
# 初始化预处理器
preprocessor = SimplePreprocessor(
    target_spacing=[1.0, 1.0, 1.0],
    normalization_scheme="min-max"
)

# 处理单个CT图像
image_paths = ["path/to/ct.nii"]
data_list, _, spacing, properties = preprocessor.run_case(image_paths)
```

### 2. 多模态MRI数据处理

```python
# 初始化预处理器
preprocessor = SimplePreprocessor(
    target_spacing=[1.0, 1.0, 1.0],
    normalization_scheme="z-score",
    target_size=[256, 256]
)

# 处理多模态MRI数据
image_paths = [
    "path/to/flair.nii",
    "path/to/t1.nii",
    "path/to/t1ce.nii",
    "path/to/t2.nii"
]
seg_path = "path/to/seg.nii"

# 执行预处理
data_list, seg, spacing, properties = preprocessor.run_case(image_paths, seg_path)
```

### 3. 自动生成数据集元数据

```python
# 配置参数
root_dir = "path/to/dataset"
api_key = "your-deepseek-api-key"

# 执行完整的元数据生成流程
analyze_directory(root_dir, sample_folder_count=5)
generate_metadata(root_dir, api_key)
execute_metadata_script(root_dir)
```

## 注意事项

1. 确保输入数据格式正确（支持.nii格式）
2. 检查磁盘空间是否充足（预处理后的数据可能较大）
3. 监控内存使用（处理大型数据集时）
4. 合理设置并行处理的进程数
5. 备份原始数据
