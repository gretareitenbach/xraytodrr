# 3D Medical Image Preprocessing Pipeline

This repository provides a set of tools and a complete workflow to preprocess 3D CT scans and their corresponding segmentations. The primary goal is to prepare medical imaging data for advanced analysis, such as diffeomorphic registration and pose estimation, using the libraries [XVR](https://www.google.com/search?q=https://github.com/eigenvivek/xvr) and [DiffPose](https://github.com/eigenvivek/diffpose) by Vivek Kumar (`eigenvivek`).

## ⚙️ Dependencies

Before starting, ensure you have the following dependencies installed:

  * **SimpleITK**: `pip install SimpleITK`
  * **NumPy**: `pip install numpy`
  * **3D Slicer**: A free, open-source platform for medical image analysis. [Download here](https://www.slicer.org/).

-----

## Workflow: Preprocessing

The preprocessing workflow is divided into four main steps. Follow them in order to ensure the data is correctly prepared.

### Step 1: Clean Segmentation Masks

**Purpose:** Raw segmentation masks often contain small, disconnected "island" artifacts. This initial step cleans the masks by isolating only the **largest connected component** for each anatomical label. This ensures that each label represents a single, contiguous object.

**Script:** `segmentation_preprocessing.py`

**Usage:**
Run the script from your terminal, providing the path to your input segmentation file and an output directory. You can optionally specify which labels to process.

```bash
python segmentation_preprocessing.py \
    --segmentation_path /path/to/your/patientA_seg.nii.gz \
    --output_folder_path /path/to/output/cleaned_segmentations \
    --labels 1 2 3 4
```

-----

### Step 2: Crop CT Scans Based on Segmentation

**Purpose:** To reduce file size and focus computational efforts, this step crops the large original CT volume to a smaller bounding box centered around the anatomy of interest (defined by the cleaned segmentation). A specified amount of padding is added around the bounding box. The script also handles mirroring for right-sided anatomy to a common (e.g., left-sided) orientation.

**Script:** `crop_cts.py`

**Usage:**
This script requires the base directory, the original CT filename, and the cleaned segmentation filename from Step 1.

```bash
python crop_cts.py \
    --base_dir /path/to/patient/data \
    --ct_filename P01_ct.nii.gz \
    --seg_filename cleaned_segmentations/P01_seg_cleaned.nii.gz \
    --padding 10.0
```

-----

### Step 3: Co-Register Cropped CTs

**Purpose:** After cropping, each CT scan exists in its own coordinate space. This step aligns, or **co-registers**, all cropped CT scans to a common orientation and position. One scan is chosen as a `reference` (or fixed) image, and all other scans are transformed to match it.

**Script:** `coregister_cts.py`

**Usage:**
Provide the directory of cropped CTs from Step 2, an output directory, and specify one of the scans to act as the reference.

```bash
python coregister_cts.py \
    --input_dir /path/to/patient/data/femur_cts_cropped \
    --output_dir /path/to/output/coregistered_cts \
    --reference_file /path/to/patient/data/femur_cts_cropped/reference_scan.nii.gz
```

-----

### Step 4: Center and Create Isometric Voxels (Manual)

**Purpose:** The final step ensures that all images are centered in their image grid and that their voxels are **isometric** (i.e., have the same physical size in all dimensions, like 1mm x 1mm x 1mm). This is crucial for many deep learning and registration algorithms.

**Script:** *None at this time. A Python script to automate this step is planned for a future release.*

**Manual Instructions using 3D Slicer:**

1.  **Load Data**: Open one of your co-registered CT scans from Step 3 in 3D Slicer.
2.  **Center the Volume**:
      * Go to the `Transforms` module.
      * Create a new linear transform (e.g., `CenteringTransform`).
      * With the transform selected, scroll down to the "Tools" section and click the **"Center volume"** button to snap the transform's center to the geometric center of your loaded CT.
      * Under the "Apply transform" section, select your CT scan as the "Transformable" and harden the transform by clicking the apply button.
3.  **Resample to Isometric Voxels**:
      * Go to the `Resample Scalar Volume` module.
      * Select your centered CT as the "Input Volume".
      * For "Output Volume", choose "Create new Volume".
      * Under "Resampling Parameters", check the **"Isotropic Spacing"** box and set the desired voxel size (e.g., `1.0` mm).
      * Click **Apply**.
4.  **Save the Result**: Save the newly created, centered, and isometric volume. Repeat this process for all co-registered CTs.
