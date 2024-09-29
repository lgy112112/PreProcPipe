import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm  # 进度条库
import multiprocessing as mp

class SimplePreprocessor:
    def __init__(self, target_spacing=[1.0, 1.0, 1.0], normalization_scheme="z-score", target_size=None):
        """
        初始化预处理器。

        参数：
        - target_spacing: 目标体素大小（spacing），默认为 [1.0, 1.0, 1.0]。
        - normalization_scheme: 归一化方案，支持 "z-score" 或 "min-max"。
        - target_size: 目标尺寸，例如 [256, 256]，默认为 None（不调整尺寸）。
        """
        self.target_spacing = target_spacing
        self.normalization_scheme = normalization_scheme
        self.target_size = target_size  # 目标大小，例如 [256, 256]

    def read_images(self, image_path):
        """
        读取图像数据 (.nii) 文件并转换为 NumPy 数组。
        """
        print("Step 1: Loading image data...")
        img = nib.load(image_path)
        img_data = img.get_fdata()
        img_spacing = img.header.get_zooms()  # 获取图像的 spacing
        print()
        return img_data, img_spacing

    def read_seg(self, seg_path):
        """
        读取分割数据 (.nii) 文件并转换为 NumPy 数组。
        """
        print("Step 1: Loading segmentation data...")
        seg = nib.load(seg_path)
        seg_data = seg.get_fdata()
        print()
        return seg_data

    def run_case(self, image_path, seg_path=None):
        """
        运行预处理流程：
        1. 读取图像与分割数据
        2. 裁剪无效区域
        3. 归一化图像数据
        4. 重采样图像和分割数据到目标体素大小
        5. 调整图像和分割数据到目标尺寸
        """
        # Step 1: 加载图像数据
        data, spacing = self.read_images(image_path)

        # 加载分割数据（如果存在）
        if seg_path:
            seg = self.read_seg(seg_path)
        else:
            seg = None

        # 打印原始数据形状
        print(f"Original image shape: {data.shape}")
        if seg is not None:
            print(f"Original segmentation shape: {seg.shape}")
        print()

        # Step 2: 裁剪无效区域
        print("Step 2: Cropping to non-zero regions...")
        data, seg, properties = self.crop_to_nonzero(data, seg)
        print()

        # 在 properties 中记录原始 spacing
        properties['original_spacing'] = spacing

        # Step 3: 归一化图像数据
        print("Step 3: Normalizing image data...")
        data = self._normalize(data, seg)
        print()

        # Step 4: 重采样到目标体素大小
        print("Step 4: Resampling data to target spacing...")
        new_shape = self.compute_new_shape(data.shape, spacing, self.target_spacing)
        data = self.resample_data(data, new_shape, order=3)  # 三次插值（图像数据）
        if seg is not None:
            seg = self.resample_data(seg, new_shape, order=0)  # 最近邻插值（分割数据）
        print()

        # Step 5: 调整到目标尺寸（如果指定）
        if self.target_size is not None:
            print("Step 5: Resizing data to target size...")
            data = self.resize_to_target_size(data, self.target_size)
            if seg is not None:
                seg = self.resize_to_target_size(seg, self.target_size, order=0)  # 分割数据使用最近邻插值
            print()

        # 返回处理后的数据和属性
        print("Preprocessing completed.\n")
        return data, seg, spacing, properties

    def crop_to_nonzero(self, data, seg):
        """
        裁剪图像和分割数据的全零区域，返回裁剪后的数据，并记录裁剪信息。
        """
        nonzero_coords = np.argwhere(data != 0)
        if nonzero_coords.size == 0:
            return data, seg, {'shape_before_cropping': data.shape,
                               'shape_after_cropping': data.shape,
                               'bbox': None}

        bbox_min = nonzero_coords.min(axis=0)
        bbox_max = nonzero_coords.max(axis=0) + 1  # 加 1 因为切片操作是排他性的

        # 裁剪数据
        cropped_data = data[bbox_min[0]:bbox_max[0],
                            bbox_min[1]:bbox_max[1],
                            bbox_min[2]:bbox_max[2]]
        if seg is not None:
            cropped_seg = seg[bbox_min[0]:bbox_max[0],
                              bbox_min[1]:bbox_max[1],
                              bbox_min[2]:bbox_max[2]]
        else:
            cropped_seg = None

        properties = {
            'shape_before_cropping': data.shape,
            'shape_after_cropping': cropped_data.shape,
            'bbox': (bbox_min.tolist(), bbox_max.tolist())
        }

        return cropped_data, cropped_seg, properties

    def _normalize(self, data, seg=None):
        """
        归一化图像数据。
        """
        if self.normalization_scheme == "z-score":
            mean_val = np.mean(data[data > 0])
            std_val = np.std(data[data > 0])
            data = (data - mean_val) / (std_val + 1e-8)
        elif self.normalization_scheme == "min-max":
            min_val = np.min(data[data > 0])
            max_val = np.max(data[data > 0])
            data = (data - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization scheme: {self.normalization_scheme}")
        return data

    def compute_new_shape(self, old_shape, old_spacing, new_spacing):
        """
        根据原始分辨率和目标分辨率计算新的形状。
        """
        resize_factor = [old_spacing[i] / new_spacing[i] for i in range(len(old_spacing))]
        new_shape = [int(np.round(old_shape[i] * resize_factor[i])) for i in range(len(old_shape))]
        print(f"Computed new shape: {new_shape}")
        return new_shape

    def resample_data(self, data, new_shape, order=3):
        """
        根据新的形状进行重采样。
        """
        print("Resampling data...")
        zoom_factors = [new_shape[i] / data.shape[i] for i in range(len(data.shape))]
        resampled_data = zoom(data, zoom_factors, order=order)
        print(f"Data resampled to shape: {resampled_data.shape}")
        return resampled_data

    def resize_to_target_size(self, data, target_size, order=3):
        """
        将图像或分割数据调整到目标尺寸。
        """
        print("Resizing data to target size...")
        current_shape = data.shape
        zoom_factors = [target_size[0] / current_shape[0],  # 调整第一个维度（Y 轴，高度）
                        target_size[1] / current_shape[1],  # 调整第二个维度（X 轴，宽度）
                        1.0]  # Z 轴（深度）保持不变
        resized_data = zoom(data, zoom_factors, order=order)
        print(f"Data resized to shape: {resized_data.shape}")
        return resized_data

# 新增：多进程处理函数
def process_case(args):
    """
    多进程调用的函数，用于处理单个病例。

    参数：
    - args: 一个包含 image_path 和 seg_path 的元组或列表
    """
    image_path, seg_path, preprocessor = args
    # 调用预处理器的 run_case 方法
    data, seg, spacing, properties = preprocessor.run_case(image_path, seg_path)
    # 返回处理结果，可以根据需要修改
    return data, seg, spacing, properties

