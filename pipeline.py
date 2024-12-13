import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from multiprocessing import Pool
import os
import csv

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

    def read_images(self, image_paths):
        """
        读取多个模态的图像数据 (.nii) 文件，并返回一个列表，每个元素为单独的 NumPy 数组。
        """
        print("Step 1: Loading multi-modal image data...")
        img_list = []
        for path in image_paths:
            img = nib.load(path)
            img_data = img.get_fdata()
            img_list.append(img_data)
        # 假设所有模态具有相同的 spacing
        img_spacing = nib.load(image_paths[0]).header.get_zooms()
        print()
        return img_list, img_spacing

    def read_seg(self, seg_path):
        """
        读取分割数据 (.nii) 文件并转换为 NumPy 数组。
        """
        print("Step 1: Loading segmentation data...")
        seg = nib.load(seg_path)
        seg_data = seg.get_fdata()
        print()
        return seg_data

    def run_case(self, image_paths, seg_path=None):
        """
        能够处理多模态图像的预处理流程，但不将它们合并到同一个数组中。
        """
        # Step 1: 加载多模态图像数据
        data_list, spacing = self.read_images(image_paths)

        if seg_path:
            seg = self.read_seg(seg_path)
        else:
            seg = None

        # 打印原始数据形状
        for i, data in enumerate(data_list):
            print(f"Original image shape (modality {i}): {data.shape}")
        if seg is not None:
            print(f"Original segmentation shape: {seg.shape}")
        print()

        # Step 2: 根据所有模态数据的非零区域计算裁剪范围
        print("Step 2: Cropping to non-zero regions...")
        # 将所有模态的非零坐标合并计算公共裁剪区域
        data_list, seg, properties = self.crop(data_list, seg)
        properties['original_spacing'] = spacing

        # Step 3: 对每个模态独立归一化
        print("Step 3: Normalizing image data...")
        for i in range(len(data_list)):
            data_list[i] = self._normalize_single_modality(data_list[i])
        print()

        # Step 4: 重采样到目标分辨率
        print("Step 4: Resampling data to target spacing...")
        # 使用第一模态计算 new_shape（假设各模态 spacing 一致）
        new_shape = self.compute_new_shape(data_list[0].shape, spacing, self.target_spacing)
        data_list = [self.resample_data(d, new_shape, order=3) for d in data_list]
        if seg is not None:
            seg = self.resample_data(seg, new_shape, order=0)
        print()

        # Step 5: 调整到目标尺寸（如果指定）
        if self.target_size is not None:
            print("Step 5: Resizing data to target size...")
            data_list = [self.resize_to_target_size(d, self.target_size, order=3) for d in data_list]
            if seg is not None:
                seg = self.resize_to_target_size(seg, self.target_size, order=0)
            print()

        print("Preprocessing completed.\n")
        return data_list, seg, spacing, properties
        

    def crop(self, data_list, seg):
        """
        裁剪图像和分割数据在 Z 轴方向的全零区域，返回裁剪后的数据列表和分割数据，以及裁剪属性。
        
        参数：
        - data_list: 多模态图像数据列表，每个元素为 NumPy 数组。
        - seg: 分割数据（NumPy 数组），可以为 None。
        
        返回：
        - cropped_data_list: 裁剪后的多模态图像数据列表。
        - cropped_seg: 裁剪后的分割数据（如果 seg 为 None，则返回 None）。
        - properties: 裁剪过程的属性信息，包括裁剪前后的形状和裁剪边界。
        """
        print("Step 2: Cropping to non-zero regions along Z-axis...")

        # 获取所有模态在 Z 轴方向的非零范围
        nonzero_slices = []
        for data in data_list:
            # 沿 Z 轴求和，如果某切片全为零，则和为零
            z_nonzero = np.any(data != 0, axis=(0, 1))
            nonzero_slices.append(np.argwhere(z_nonzero).flatten())

        if len(nonzero_slices) == 0:
            # 全部为零，不裁剪
            properties = {
                'shape_before_cropping': [d.shape for d in data_list],
                'shape_after_cropping': [d.shape for d in data_list],
                'z_bbox': None
            }
            return data_list, seg, properties

        # 计算公共 Z 轴范围
        z_min = min(s.min() for s in nonzero_slices)
        z_max = max(s.max() for s in nonzero_slices) + 1  # 加1表示包含该索引

        print(f"Z-axis cropping range: {z_min} to {z_max}")

        # 裁剪所有模态的 Z 轴范围
        cropped_data_list = [d[:, :, z_min:z_max] for d in data_list]

        # 裁剪分割数据的 Z 轴范围
        cropped_seg = None
        if seg is not None:
            cropped_seg = seg[:, :, z_min:z_max]

        # 记录裁剪属性
        properties = {
            'shape_before_cropping': [d.shape for d in data_list],
            'shape_after_cropping': [d.shape for d in cropped_data_list],
            'z_bbox': (z_min, z_max)
        }

        print(f"Shapes before cropping: {[d.shape for d in data_list]}")
        print(f"Shapes after cropping: {[d.shape for d in cropped_data_list]}")
        if seg is not None:
            print(f"Segmentation shape after cropping: {cropped_seg.shape}")

        return cropped_data_list, cropped_seg, properties


    # def _normalize(self, data, seg=None):
    #     """
    #     归一化图像数据。
    #     """
    #     if self.normalization_scheme == "z-score":
    #         mean_val = np.mean(data[data > 0])
    #         std_val = np.std(data[data > 0])
    #         data = (data - mean_val) / (std_val + 1e-8)
    #     elif self.normalization_scheme == "min-max":
    #         min_val = np.min(data[data > 0])
    #         max_val = np.max(data[data > 0])
    #         data = (data - min_val) / (max_val - min_val + 1e-8)
    #     else:
    #         raise ValueError(f"Unknown normalization scheme: {self.normalization_scheme}")
    #     return data

    # 新增一个专门处理单个模态归一化的方法
    def _normalize_single_modality(self, data):
        """
        对单个模态数据进行归一化。
        """
        mask = data > 0
        if self.normalization_scheme == "z-score":
            mean_val = np.mean(data[mask]) if np.any(mask) else 0.0
            std_val = np.std(data[mask]) if np.any(mask) else 1.0
            data = (data - mean_val) / (std_val + 1e-8)
        elif self.normalization_scheme == "min-max":
            min_val = np.min(data[mask]) if np.any(mask) else 0.0
            max_val = np.max(data[mask]) if np.any(mask) else 1.0
            data = (data - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization scheme: {self.normalization_scheme}")
        return data

    def compute_new_shape(self, old_shape, old_spacing, new_spacing):
        """
        根据原始分辨率和目标分辨率计算新的形状。
        """
        resize_factor = [old_spacing[i] / new_spacing[i] for i in range(len(old_spacing))]
        print(f"Computed resize factors: {resize_factor}")
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


def process_case(args):
    """
    多进程调用的函数，用于处理单个病例。

    参数：
    - args: (sample_id, image_paths, seg_path, preprocessor, output_root)
    """
    sample_id, image_paths, seg_path, preprocessor, output_root = args
    # 调用预处理器的 run_case 方法处理多模态图像
    data_list, seg, spacing, properties = preprocessor.run_case(image_paths, seg_path)

    # 创建样本目录（在output_root下）
    sample_dir = os.path.join(output_root, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    # 推断各模态名称（使用文件名去除扩展名作为模态名称）
    modality_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    # 保存各模态数据
    modality_paths = []
    for modality_name, modality_data in zip(modality_names, data_list):
        save_path = os.path.join(sample_dir, f"{modality_name}.npz")
        np.savez_compressed(save_path, data=modality_data)
        modality_paths.append(save_path)

    seg_path_out = None
    # 保存分割数据（如果有分割）
    if seg is not None:
        seg_save_path = os.path.join(sample_dir, "seg.npz")
        np.savez_compressed(seg_save_path, data=seg)
        seg_path_out = seg_save_path

    # 保存 spacing 和 properties 为 meta.npz
    meta_save_path = os.path.join(sample_dir, "meta.npz")
    np.savez_compressed(meta_save_path, spacing=spacing, properties=properties)

    # 返回处理结果及保存的文件路径信息，用于后续生成metadata.csv
    return {
        "sample_id": sample_id,
        "modality_paths": modality_paths,
        "seg_path": seg_path_out,
        "meta_path": meta_save_path
    }


def run_in_parallel(preprocessor, cases, output_root, num_workers=4):
    """
    使用多进程并行处理多个病例，并在output_root下存放处理结果为npz文件，
    同时在output_root下生成metadata.csv记录每个sample的npz地址。

    参数：
    - preprocessor: SimplePreprocessor 实例。
    - cases: 包含多个病例信息的列表，每个病例是一个字典，格式：
        {
            "sample_id": "某病例ID字符串",
            "image_paths": [模态1路径, 模态2路径, ...],
            "seg_path": 分割路径或 None
        }
    - output_root: 输出结果保存的根目录
    - num_workers: 并行进程数，默认为 4。

    返回：
    - results: 包含每个病例保存文件路径信息的列表
    """
    os.makedirs(output_root, exist_ok=True)

    args_list = [
        (case["sample_id"], case["image_paths"], case["seg_path"], preprocessor, output_root) for case in cases
    ]

    # 使用多进程池并行处理
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_case, args_list)

    # 生成 metadata.csv
    # 文件内容格式示例：
    # sample_id,modality_paths,seg_path,meta_path
    # BraTS2021_00000,"['output_root/BraTS2021_00000/t1.npz','output_root/BraTS2021_00000/t2.npz']","output_root/BraTS2021_00000/seg.npz","output_root/BraTS2021_00000/meta.npz"

    csv_path = os.path.join(output_root, "metadata.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sample_id", "modality_paths", "seg_path", "meta_path"])
        for res in results:
            # 将绝对路径转换为相对于output_root的相对路径，便于移植
            # 如果需要保留绝对路径，可注释掉此步骤
            rel_modality_paths = [os.path.relpath(p, output_root) for p in res["modality_paths"]]
            rel_seg_path = os.path.relpath(res["seg_path"], output_root) if res["seg_path"] is not None else None
            rel_meta_path = os.path.relpath(res["meta_path"], output_root)
            writer.writerow([
                res["sample_id"],
                str(rel_modality_paths),
                rel_seg_path,
                rel_meta_path
            ])

    return results


