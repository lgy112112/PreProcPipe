
# PreProcPipe: A Multi-Modal Image Preprocessing Pipeline for CT/MRI

## Choose Language / 选择语言

- [English](readme_en.md)
- [简体中文](readme.md)

## 1. PreProcPipe Project Structure

This project demonstrates a preprocessing pipeline using the BraTS2021 dataset as an example. The main files and directory structure are as follows:

### Data Directory

`PreProcPipe/BraTS2021_Training_Data`

Contains the original training data for the BraTS2021 dataset. Each sample has its own folder identified by the sample's ID.

#### Sample Directory

- `PreProcPipe/BraTS2021_Training_Data/BraTS2021_00000/`
  - `BraTS2021_00000_flair/`
    - Contains the FLAIR modality file for this sample, e.g., `00000057_brain_flair.nii`.
  - `BraTS2021_00000_seg/`
    - Contains the segmentation file for this sample.
  - `BraTS2021_00000_t1/`
    - Contains the T1 modality file for this sample.
  - `BraTS2021_00000_t1ce/`
    - Contains the T1CE modality file for this sample.
  - `BraTS2021_00000_t2/`
    - Contains the T2 modality file for this sample.

#### Other Samples

Similar structures apply to other samples, for example:

- `PreProcPipe/BraTS2021_Training_Data/BraTS2021_00002/`
- `PreProcPipe/BraTS2021_Training_Data/BraTS2021_00003/`

### Code and Tutorials

- `PreProcPipe/tutorial copy.py`
  - Contains a detailed tutorial on using `pipeline.py` to process data, demonstrating how to load, preprocess, and save results. It's recommended to start with this tutorial.
- `PreProcPipe/pipeline.py`
  - The main preprocessing script that contains the logic for cropping, normalizing, and resampling BraTS2021 data.

---

## 2. Code Highlights

The `SimplePreprocessor` class is the core component for multi-modal image preprocessing. It handles multi-modal MRI or CT data and performs preprocessing steps like cropping, normalization, resampling, and resizing.

### Preprocessing Steps and Methods

#### 1. Initialization

Configures preprocessing parameters using the `__init__` method:

-   **`target_spacing`**: Specifies the target voxel size (default: `[1.0, 1.0, 1.0]`).
-   **`normalization_scheme`**: Specifies the normalization method (`z-score` or `min-max`).
-   **`target_size`**: Specifies the target size (e.g., `[256, 256]`), defaults to `None` (no resizing).

#### 2. Data Loading

-   **`read_images(image_paths)`**: Loads multi-modal image data, returning a list of NumPy arrays and voxel spacing.
-   **`read_seg(seg_path)`**: Loads segmentation data, returning a NumPy array.

#### 3. Cropping

-   **`crop(data_list, seg)`**:
    -   Crops only the all-zero regions along the Z-axis.
    -   Returns the cropped image data, segmentation data, and cropping properties (cropping range and shape changes).

#### 4. Normalization

-   **`_normalize_single_modality(data)`**:
    -   Normalizes data for a single modality.
    -   Supports `z-score` and `min-max` normalization methods.

#### 5. Resampling

-   **`compute_new_shape(old_shape, old_spacing, new_spacing)`**:
    -   Calculates the target shape based on the original shape and voxel spacing.
    -   Outputs the resampling factor and new shape.
-   **`resample_data(data, new_shape, order=3)`**:
    -   Resamples the image data to the target shape.
    -   Uses cubic interpolation by default.

#### 6. Resizing

-   **`resize_to_target_size(data, target_size, order=3)`**:
    -   Resizes image data to the specified target size (e.g., `[256, 256]`).
    -   Keeps the Z-axis depth unchanged by default.

#### 999. Run Preprocessing for a Single Sample - The Combination of Above Methods

-   **`run_case(image_paths, seg_path=None)`**:
    Executes the following steps in order:
    1.  **Data Loading**: Loads all modal images and corresponding segmentation data.
    2.  **Z-Axis Cropping**: Calls the `crop` method to crop only the all-zero regions in the Z-axis, keeping other dimensions.
    3.  **Normalization**: Independently normalizes data for each modality (`_normalize_single_modality` method).
    4.  **Resampling**: Adjusts the voxel resolution using `compute_new_shape` and `resample_data`.
    5.  **Resizing**: Adjusts the size of the data based on target dimensions (`resize_to_target_size` method).
    6.  **Return Results**: Outputs the cropped data, segmentation data, original resolution information, and cropping attributes.

### Processing Flow Overview

The `SimplePreprocessor` is designed to handle multi-modal medical imaging data with segmentation data. The functionality of each step is modularized, making it easy to extend and reuse. It supports the majority of common preprocessing needs for high-dimensional data.

---

## 3. How to Use

`SimplePreprocessor` offers a flexible interface suitable for both multi-modal and single-modal image data preprocessing. Below are the specific instructions on how to use it:

### Input Data Format

#### Multi-Modal Data

For multi-modal data (e.g., FLAIR, T1, T1CE, T2), the input data should be a list of file paths pointing to the `.nii` files for each modality. For example:

```python
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii",
    "BraTS2021_00000/BraTS2021_00000_t1/00000057_brain_t1.nii",
    "BraTS2021_00000/BraTS2021_00000_t1ce/00000057_brain_t1ce.nii",
    "BraTS2021_00000/BraTS2021_00000_t2/00000057_brain_t2.nii"
]
```

#### Single-Modal Data

For single-modal data, the input data only needs to contain the path to a single `.nii` file, for example:

```python
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii"
]
```

#### Segmentation Data

The input for segmentation data is a single file path pointing to the `.nii` format segmentation file. For example:

```python
seg_path = "BraTS2021_00000/BraTS2021_00000_seg/00000057_seg.nii"
```

Segmentation data is optional. If there is no segmentation data, set `seg_path` to `None`.

---

### How to Call the Preprocessing Method

#### 1. Initialize the Preprocessor

First, create an instance of `SimplePreprocessor`. You can specify the following parameters:

-   `target_spacing`: The target voxel size (defaults to `[1.0, 1.0, 1.0]`).
-   `normalization_scheme`: The normalization method (defaults to `"z-score"`).
-   `target_size`: The target size (defaults to `None`, meaning no resizing).

For example:

```python
from pipeline import SimplePreprocessor

preprocessor = SimplePreprocessor(
    target_spacing=[1.0, 1.0, 1.0],
    normalization_scheme="z-score",
    target_size=[256, 256]
)
```

---

#### 2. Run Preprocessing for a Single Sample

Use the `run_case` method to preprocess a single sample:

```python
# Input data
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii",
    "BraTS2021_00000/BraTS2021_00000_t1/00000057_brain_t1.nii",
    "BraTS2021_00000/BraTS2021_00000_t1ce/00000057_brain_t1ce.nii",
    "BraTS2021_00000/BraTS2021_00000_t2/00000057_brain_t2.nii"
]
seg_path = "BraTS2021_00000/BraTS2021_00000_seg/00000057_seg.nii"

# Run preprocessing
data_list, seg, spacing, properties = preprocessor.run_case(image_paths, seg_path)
```

-   `data_list`: Preprocessed multi-modal image data (after cropping, normalization, resampling, and resizing).
-   `seg`: Preprocessed segmentation data (if available).
-   `spacing`: Voxel spacing information of the original image.
-   `properties`: Attributes related to cropping and preprocessing (e.g., shapes before and after cropping, cropping boundaries).

---

#### 3. Run Preprocessing for a Single-Modal Sample

For single-modal data, the input list contains only one file path:

```python
image_paths = [
    "BraTS2021_00000/BraTS2021_00000_flair/00000057_brain_flair.nii"
]
seg_path = None  # If there is no segmentation data

data_list, seg, spacing, properties = preprocessor.run_case(image_paths, seg_path)
```

In this case:

-   `data_list` contains the processing results of a single modality.
-   `seg` is `None`.

---

### Batch Processing Samples

If you need to process multiple samples, you can store the input of each sample (`image_paths` and `seg_path`) in a list and use multi-processing tools (such as the `run_in_parallel` method).

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

# Batch processing
results = run_in_parallel(preprocessor, cases, num_workers=4, output_root="preprocessed_data")
```

`results` returns the preprocessing results for each sample. The data will also be saved to the corresponding folders as set by the argument `output_root="preprocessed_data"`.

### Output Data

After processing, the return value for each sample includes:

1.  **`data_list`**: A list storing the processed data for each modality.
2.  **`seg`**: The processed segmentation data (if available).
3.  **`spacing`**: The original voxel spacing.
4.  **`properties`**: Records the information for cropping, normalization, and resampling, for example:

    ```python
    {
        "shape_before_cropping": [(240, 240, 155), ...],
        "shape_after_cropping": [(240, 240, 120), ...],
        "z_bbox": [10, 130]
    }
    ```

With this structured return value, you can easily save or analyze the results.
```
