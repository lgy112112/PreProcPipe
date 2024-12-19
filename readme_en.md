
# PreProcPipe: A Multi-Modal Image Preprocessing Pipeline for CT/MRI

![xmp](before.png)
![xmp](after.png)

<p align="center">
  <a href="images/alipay-qrcode.jpg">
    <img src="https://img.shields.io/badge/%E6%8A%95%E5%96%82%E4%B8%BB%E6%92%AD-%E5%8D%95%E5%87%BB%E6%94%AF%E6%8C%81-9cf?style=for-the-badge&logo=alipay" alt="helpmeiamsofuckinghungry">
  </a>
  <a href="images/alipay-qrcode.jpg">
    <img src="https://img.shields.io/badge/%E6%8A%95%E5%96%82%E4%B8%BB%E6%92%AD-%E5%8D%95%E5%87%BB%E6%94%AF%E6%8C%81-9cf?style=for-the-badge&logo=alipay" alt="helpmeiamsofuckinghungry">
  </a>
  <a href="images/alipay-qrcode.jpg">
    <img src="https://img.shields.io/badge/%E6%8A%95%E5%96%82%E4%B8%BB%E6%92%AD-%E5%8D%95%E5%87%BB%E6%94%AF%E6%8C%81-9cf?style=for-the-badge&logo=alipay" alt="helpmeiamsofuckinghungry">
  </a>
  <a href="images/alipay-qrcode.jpg">
    <img src="https://img.shields.io/badge/%E6%8A%95%E5%96%82%E4%B8%BB%E6%92%AD-%E5%8D%95%E5%87%BB%E6%94%AF%E6%8C%81-9cf?style=for-the-badge&logo=alipay" alt="helpmeiamsofuckinghungry">
  </a>
  <a href="images/alipay-qrcode.jpg">
    <img src="https://img.shields.io/badge/%E6%8A%95%E5%96%82%E4%B8%BB%E6%92%AD-%E5%8D%95%E5%87%BB%E6%94%AF%E6%8C%81-9cf?style=for-the-badge&logo=alipay" alt="helpmeiamsofuckinghungry">
  </a>
</p>

## Choose Language / 选择语言

- [English](readme_en.md)
- [简体中文](readme.md)

## Updates

### -2. `2024/12/19` Update
**Bug Fix:** A bug in `pipeline.py` that prevented the recognition of integer-type `sample_id` values, thus causing file saving failures, has been fixed.

### -1. `2024/12/18` Update
**New Feature:** Added `pipeline_example.py`. This new file demonstrates how to use `pipeline.py` for data preprocessing with the `metadata.csv` file obtained from `LLM_metadata.py`. Although a small bug remains, it doesn't hinder functionality.

### 0. Initial Update
**New Feature:** You can now automatically generate `metadata.csv` using an LLM!

This update introduces a fully automated process that uses **LLM (Large Language Model)** to automatically generate the `metadata.csv` file for medical image datasets. This process simplifies data preprocessing and significantly reduces manual effort and errors. Here are the core steps:

---

### 1. Directory Analysis and Sampling

First, the `analyze_directory` function analyzes and samples the dataset's directory structure. It traverses the root directory, identifies all Level A folders (sample folders), and randomly samples files from these folders and their subfolders. The results are saved as a JSON file named `directory_analysis.json`.

#### Key Features:
*   **Folder Structure Analysis:** Recursively traverses the folder structure to create a directory tree.
*   **Random Sampling:** Randomly samples a defined number of files from each Level A folder.
*   **Result Saving:** Saves the analysis as `directory_analysis.json`, for subsequent steps.

#### How to Use:
```python
analyze_directory(root_directory, sample_folder_count=5, sample_file_count=10)
```
*   `root_directory`: The path to the root directory of the dataset.
*   `sample_folder_count`: The number of Level A folders to randomly sample.
*   `sample_file_count`: The number of files to randomly sample from each folder.

---

### 2. LLM-Generated Python Code for `metadata.csv`

Next, the `generate_metadata` function takes `directory_analysis.json` as input and uses the DeepSeek API to generate Python code for building `metadata.csv`. The LLM automatically identifies multimodal files and mask files based on file naming patterns and generates the appropriate code.

#### Key Features:
*   **File Naming Pattern Analysis:** The LLM analyzes file names to identify multimodal and mask files.
*   **Code Generation:** Generates Python code for building the `metadata.csv` file.
*   **Result Saving:** Saves the generated code as `generate_metadata.py`.

#### How to Use:
```python
generate_metadata(root_directory, your_api_key=None)
```
*   `root_directory`: The path to the root directory of the dataset.
*   `your_api_key`: Your DeepSeek API key (you can get it for free from [DeepSeek's official website](https://platform.deepseek.com/usage)).

---

### 3. Executing the Code and Generating `metadata.csv`

Finally, the `execute_metadata_script` function executes the generated `generate_metadata.py` script to automatically create `metadata.csv`. The function checks if the CSV file is created correctly and prints its first 5 rows for verification.

#### Key Features:
*   **Code Execution:** Executes the `generate_metadata.py` script to create the `metadata.csv` file.
*   **Result Verification:** Checks if the CSV file exists and prints the first 5 rows.

#### How to Use:
```python
execute_metadata_script(root_directory)
```
*   `root_directory`: The path to the root directory of the dataset.

---

### Summary

Through these three steps, you can easily generate the `metadata.csv` using LLM, without manual coding or analyzing file naming patterns. This fully automated process is suitable for various medical image datasets.

#### Usage Example:
```python
if __name__ == "__main__":
    # 1. Analyze the directory structure
    root_directory = "/teamspace/studios/this_studio/kaggle_3m"  # Fill in the root directory of the dataset (must be an absolute path)
    analyze_directory(root_directory=root_directory, sample_folder_count=5, sample_file_count=10)

    # 2. Generate Python code for metadata.csv
    generate_metadata(root_directory=root_directory, your_api_key=None)

    # 3. Execute the code and check metadata.csv
    execute_metadata_script(root_directory=root_directory)
```

---

### Notes

*   Ensure your dataset files have consistent naming patterns, so the LLM can correctly identify multimodal and mask files.
*   For larger datasets, increase `sample_folder_count` and `sample_file_count` for better LLM analysis accuracy.
*   If you use the DeepSeek API, make sure you have obtained your API key and included it in `LLM_metadata.py`.

---

## 1. PreProcPipe Project Structure

This project uses the BraTS2021 dataset for preprocessing examples. The primary files and directories are:

### Data Directory
`PreProcPipe/BraTS2021_Training_Data`
Contains the original BraTS2021 training data, with each sample stored in its own folder using its ID.

#### Sample Directories
*   `PreProcPipe/BraTS2021_Training_Data/BraTS2021_00000/`
    *   `BraTS2021_00000_flair/`
        *   Contains the FLAIR modality file, e.g., `00000057_brain_flair.nii`
    *   `BraTS2021_00000_seg/`
        *   Contains the segmentation file.
    *   `BraTS2021_00000_t1/`
        *   Contains the T1 modality file.
    *   `BraTS2021_00000_t1ce/`
        *   Contains the T1CE modality file.
    *   `BraTS2021_00000_t2/`
        *   Contains the T2 modality file.

#### Other Samples
Similar structures for other samples, e.g.:
*   `PreProcPipe/BraTS2021_Training_Data/BraTS2021_00002/`
*   `PreProcPipe/BraTS2021_Training_Data/BraTS2021_00003/`

### Code and Tutorials
*   `PreProcPipe/tutorial.ipynb`
    *   A detailed tutorial on using `pipeline.py`, demonstrating how to load, preprocess, and save data. Recommended for new users.
*   `PreProcPipe/pipeline.py`
    *   Main preprocessing script containing the code logic for cropping, normalizing, and resampling BraTS2021 data.
*   `PreProcPipe/LLM_metadata.py`
    *   A script for using an LLM to generate `metadata.csv`.
*   `PreProcPipe/How_I_Use_LLM_to_DIY_metadata.ipynb`
    *   A notebook documenting the steps for using an LLM to get `metadata.csv`.

---

## 2. Code Highlights

The `SimplePreprocessor` is the core class for multimodal image preprocessing, designed to handle multimodal MRI or CT data. It performs cropping, normalization, resampling, and resizing.

### Preprocessing Steps and Methods

#### 1. Initialization
The `__init__` method configures the preprocessing parameters:

*   `target_spacing`: The target voxel spacing (default `[1.0, 1.0, 1.0]`).
*   `normalization_scheme`: Normalization method (`z-score` or `min-max`).
*   `target_size`: The target image size (e.g., `[256, 256]`, defaults to `None` for no resizing).

#### 2. Data Loading
*   `read_images(image_paths)`: Loads multimodal image data, returning a list of NumPy arrays and voxel spacing.
*   `read_seg(seg_path)`: Loads segmentation data, returning a NumPy array.

#### 3. Cropping Functionality
*   `crop(data_list, seg)`:
    *   Crops the image and segmentation data along the Z-axis only to remove all-zero regions.
    *   Returns cropped image data, segmentation data, and cropping properties (bounding box and shape changes).

#### 4. Normalization
*   `_normalize_single_modality(data)`:
    *   Normalizes single-modality data.
    *   Supports `z-score` and `min-max` normalization methods.

#### 5. Resampling
*   `compute_new_shape(old_shape, old_spacing, new_spacing)`:
    *   Calculates the target shape based on the original shape and voxel spacing.
    *   Outputs the resampling factor and the new shape.
*   `resample_data(data, new_shape, order=3)`:
    *   Resamples image data to the target shape using cubic interpolation by default.

#### 6. Resizing
*   `resize_to_target_size(data, target_size, order=3)`:
    *   Resizes image data to the specified target size (e.g., `[256, 256]`).
    *   Keeps the Z-axis depth unchanged by default.

#### 7. Single Sample Preprocessing - Combining All Functions Above
*   `run_case(image_paths, seg_path=None)`:
    Executes the following steps:
    1.  **Data Loading**: Loads all modality images and the corresponding segmentation data.
    2.  **Z-axis Cropping**: Calls the `crop` method to only crop Z-axis all-zero regions, preserving other dimensions.
    3.  **Normalization**: Normalizes each modality independently using `_normalize_single_modality`.
    4.  **Resampling**: Uses `compute_new_shape` and `resample_data` to adjust voxel resolution.
    5.  **Resizing**: Adjusts the data size based on the target size, using `resize_to_target_size`.
    6.  **Return Results**: Outputs the cropped data, segmentation data, original spacing info, and cropping properties.

### Process Overview

`SimplePreprocessor` is designed for preprocessing medical image data with multimodal images and segmentation. Each step is modular for easy extension and reuse, and it supports most common preprocessing needs for high-dimensional data.

---

## 3. How to Use It?

`SimplePreprocessor` provides a flexible interface for preprocessing multimodal and single-modality image data. Here's a detailed guide on how to use it:

### Input Data Format

#### Multimodal Data
For multimodal data (e.g., FLAIR, T1, T1CE, T2), the input should be a list of file paths, each pointing to a `.nii` file for a specific modality. For example:

```python
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii",
    "BraTS2021_00000/BraTS2021_00000_t1/00000057_brain_t1.nii",
    "BraTS2021_00000/BraTS2021_00000_t1ce/00000057_brain_t1ce.nii",
    "BraTS2021_00000/BraTS2021_00000_t2/00000057_brain_t2.nii"
]
```

#### Single Modality Data
For single-modality data, the input data should be a list containing only one `.nii` file path:

```python
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii"
]
```

#### Segmentation Data
Segmentation data is input as a single file path to a `.nii` segmentation file. For example:

```python
seg_path = "BraTS2021_00000/BraTS2021_00000_seg/00000057_seg.nii"
```

Segmentation data is optional. Set `seg_path` to `None` if no segmentation data is available.

---

### Calling Preprocessing Methods

#### 1. Initialize the Preprocessor
Create an instance of `SimplePreprocessor`, specifying the following parameters:

*   `target_spacing`: The target voxel spacing (default is `[1.0, 1.0, 1.0]`).
*   `normalization_scheme`: The normalization method (default is `"z-score"`).
*   `target_size`: The target size (default is `None`, which means no resizing).

Example:

```python
from pipeline import SimplePreprocessor

preprocessor = SimplePreprocessor(
    target_spacing=[1.0, 1.0, 1.0],
    normalization_scheme="z-score",
    target_size=[256, 256]
)
```

---

#### 2. Run a Single Sample
Use the `run_case` method to preprocess a single sample:

```python
# Input Data
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii",
    "BraTS2021_00000/BraTS2021_00000_t1/00000057_brain_t1.nii",
    "BraTS2021_00000/BraTS2021_00000_t1ce/00000057_brain_t1ce.nii",
    "BraTS2021_00000/BraTS2021_00000_t2/00000057_brain_t2.nii"
]
seg_path = "BraTS2021_00000/BraTS2021_00000_seg/00000057_seg.nii"

# Run the preprocessing
data_list, seg, spacing, properties = preprocessor.run_case(image_paths, seg_path)
```

*   `data_list`: Preprocessed multimodal image data (after cropping, normalization, resampling, and resizing).
*   `seg`: Preprocessed segmentation data (if available).
*   `spacing`: Voxel spacing information from the original image.
*   `properties`: Attributes related to cropping and preprocessing (e.g., shape before and after cropping, cropping boundaries).

---

#### 3. Run a Single-Modality Sample
For single-modality data, the input list contains only one file path:

```python
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii"
]
seg_path = None  # If no segmentation data

data_list, seg, spacing, properties = preprocessor.run_case(image_paths, seg_path)
```

*   `data_list` will contain the preprocessing results for the single modality.
*   `seg` will be `None`.

---

### Batch Processing Samples
To process multiple samples, store the inputs (`image_paths` and `seg_path`) for each sample in a list, and use a multiprocessing tool (e.g., `run_in_parallel`).

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

# Batch Process
results = run_in_parallel(preprocessor, cases, num_workers=4, output_root="preprocessed_data")
```

`results` returns the preprocessing results for each sample. The data will also be saved in specified the `output_root="preprocessed_data"` directory.

### Output Data
After processing, each sample's return value includes:

1.  `data_list`: A list storing the preprocessed data for each modality.
2.  `seg`: Preprocessed segmentation data (if available).
3.  `spacing`: The original voxel spacing.
4.  `properties`: Information about cropping, normalization, and resampling, for example:
    ```python
    {
        "shape_before_cropping": [(240, 240, 155), ...],
        "shape_after_cropping": [(240, 240, 120), ...],
        "z_bbox": [10, 130]
    }
    ```

This structured return allows you to easily save or analyze the results in your next steps.

---

This detailed breakdown should make it easy to understand and use the provided code. Let me know if you have any more questions.

