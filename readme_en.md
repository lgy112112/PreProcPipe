
# PreProcPipe

## Project Introduction

PreProcPipe is a pipeline designed for medical image preprocessing, inspired by the nnUNet processing workflow (nnUNet uses tedious JSON configuration, which I want to avoid). It includes core functionalities such as loading medical images, cropping, normalization, resampling, and adjusting to target sizes. This project utilizes multiprocessing for parallel processing, supports `.nii` format medical image data, and saves the processed results as `.npz` files.

## Main Features

- Read medical image data (supports `.nii` format)
- Crop the non-zero regions of the images
- Normalize images (supports z-score and min-max normalization)
- Resample images to the target voxel size
- Adjust the image and segmentation data to the target size
- Accelerate processing through parallel execution using multiprocessing

## File Structure

- `pipeit.py`: The main processing script, reads the data paths from the `metadata.csv` file and processes each case using multiprocessing.
- `pipeline.py`: Defines the `SimplePreprocessor` class, which provides the functionality for loading, preprocessing, and saving images.

## Environment Requirements

- Python 3.6+
- Required libraries:
  - `numpy`
  - `pandas`
  - `nibabel`
  - `tqdm`
  - `scipy`
  - `multiprocessing`
  - `IPython` (for clearing output)

You can install the required dependencies using the following command:
```bash
pip install numpy pandas nibabel tqdm scipy ipython
```

## How to Use

### 1. Configure `metadata.csv`

First, you need to prepare a `metadata.csv` file that contains the paths of the image and segmentation files. The `metadata.csv` file should have the following format:

| case_id  | image_path                  | label_path                  |
|----------|-----------------------------|-----------------------------|
| case_001 | /path/to/image1.nii.gz       | /path/to/label1.nii.gz       |
| case_002 | /path/to/image2.nii.gz       | /path/to/label2.nii.gz       |

- `case_id`: Unique identifier for the case.
- `image_path`: Path to the image file (in `.nii` format).
- `label_path`: Path to the segmentation file (if available).

### 2. Run the Preprocessing

To start the preprocessing pipeline, run the following command in the project directory:

```bash
python pipeit.py
```

The program will automatically read the file paths from the `metadata.csv` file and process each case sequentially.

### 3. Processing Progress

During preprocessing, the program will display the current case number and the total number of cases. For example:

```
Currently processing case 1/100: case_001
```

By utilizing multiprocessing, the program speeds up processing, and the processed image data will be saved as `.npz` files in the `processed_data` directory.

### 4. Processed Results

For each processed case, the program will generate the following two files and save them to the `processed_data` directory:

- `{case_id}_data.npz`: The processed image data.
- `{case_id}_seg.npz`: The processed segmentation label data (if available).

## Code Explanation

### pipeit.py

This script is responsible for reading the case information from the `metadata.csv` file and using multiprocessing to process each case's data.

```python
# Using pandas to read the CSV file
metadata_df = pd.read_csv('metadata.csv')

# Iterate over each case's data and process using multiprocessing
for idx, row in metadata_df.iterrows():
    process_case((idx, row))
```

The `process_case()` function performs the following tasks:
- Load image and segmentation data
- Create an instance of the preprocessor
- Perform cropping, normalization, resampling, and resizing on the data
- Save the processed data

### pipeline.py

This file contains the core preprocessing class `SimplePreprocessor`, which is responsible for processing both the image and segmentation data.

- `read_images()`: Loads the `.nii` format image files.
- `run_case()`: Executes the preprocessing steps, including cropping, normalization, resampling, and resizing.
- `resample_data()`: Resamples the image data to the target voxel size.
- `resize_to_target_size()`: Adjusts the image or segmentation data to the target size.

## Frequently Asked Questions

### 1. Incorrect `metadata.csv` Format

Make sure that the `case_id`, `image_path`, and `label_path` columns are correctly configured in the `metadata.csv` file. The paths should be the full file paths.

### 2. Running Out of Memory

If you are processing a large amount of data, consider reducing the number of parallel processes by adjusting the `num_processes` parameter in `mp.Pool(processes=num_processes)` in `pipeit.py`.

### 3. Cannot Find the Processed `.npz` Files

Ensure that the `processed_data` directory has been created during the script execution, and the `.npz` files are being saved to the correct path.

## Contributions and Feedback

If you encounter any issues or have any suggestions during use, feel free to submit issues or provide feedback on the GitHub project's issue page.

## License

This project is licensed under the MIT License. For more details, please see the LICENSE file.