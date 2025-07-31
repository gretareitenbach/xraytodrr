# 2D-3D X-Ray Registration Pipeline

This repository provides a set of tools and a complete workflow to preprocess 3D CT scans and their corresponding segmentations. The primary goal is to prepare medical imaging data for advanced analysis, such as diffeomorphic registration and pose estimation, using the libraries [XVR](https://github.com/eigenvivek/xvr) and [DiffDRR](https://github.com/eigenvivek/DiffDRR) by Vivek Gopalakrishnan (`eigenvivek`).

## Preprocessing

### Step 1: Clean Segmentation Masks

**Purpose:** Raw segmentation masks often contain small, disconnected "island" artifacts. This initial step cleans the masks by isolating only the **largest connected component** for each anatomical label. This ensures that each label represents a single, contiguous object.

**Script:** `segmentation_preprocessing.py`

**Usage:**
Run the script from your terminal, providing the path to your input segmentation file and an output directory. You can optionally specify which labels to process.

```bash
With saved defaults:

python segmentation_preprocessing.py \
    -i /path/to/your/patientA_seg.nii.gz \
    -o /path/to/your/processed_data

With specified defaults:

python segmentation_preprocessing.py \
    -i /path/to/your/patientB_seg.nii.gz \
    -o /path/to/your/processed_data \
    -labels 1 2 3
```

-----

### Step 2: Crop CT Scans Based on Segmentation

**Purpose:** To reduce file size and focus computational efforts, this step crops the large original CT volume to a smaller bounding box centered around the anatomy of interest (defined by the cleaned segmentation). A specified amount of padding is added around the bounding box. The script also handles mirroring for right-sided anatomy to a common (e.g., left-sided) orientation.

**Script:** `crop_cts.py`

**Usage:**
This script requires the base directory, the original CT filename, and the cleaned segmentation filename from Step 1.

```bash
With saved defaults:

python crop_cts.py \
    --base_dir /path/to/patient/data \
    --ct_filename P01_ct.nii.gz \
    --seg_filename segmentations/P01_seg.nii.gz

With specified defaults:

python crop_cts.py \
    --base_dir /path/to/patient/data \
    --ct_filename P01_ct.nii.gz \
    --seg_filename segmentations/P01_seg.nii.gz \
    --labels 10 11 20 21 \
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
    --input_dir /path/to/source_cts \
    --output_dir /path/to/coregistered_cts \
    --reference_file /path/to/source_cts/reference_scan.nii.gz
```

-----

### Step 4: Center and Create Isometric Voxels (Manual)

**Purpose:** The final step ensures that all images are centered in their image grid and that their voxels are **isometric** (i.e., have the same physical size in all dimensions, like 1mm x 1mm x 1mm).

**Script:** *None at this time. A Python script to automate this step is planned for a future release.*

**Manual Instructions using 3D Slicer:**

1.  **Load Data**: Open the co-registered CT scans from Step 3 in 3D Slicer.
2.  **Center the Volume**:
      * Go to the `Transforms` module.
      * Create a new linear transform (e.g., `CenteringTransform`).
      * With the transform selected, scroll down to the "Tools" section and click the **"Center volume"** button to snap the transform's center to the geometric center of your loaded CT.
      * Under the "Apply transform" section, select your CT scan as the "Transformable" and harden the transform by clicking the apply button.
3.  **Resample to Isometric Voxels**:
      * Go to the `Resample Scalar Volume` module.
      * Select your centered CT as the "Input Volume".
      * For "Output Volume", choose "Create new Volume".
      * Under "Resampling Parameters", set the desired voxel size (e.g., `1.0`x`1.0`x`1.0` mm).
      * Click **Apply**.
4.  **Save the Result**: Save the newly created, centered, and isometric volume. Repeat this process for all co-registered CTs.

-----

## Model Training

First, a general **agnostic model** is trained on the entire dataset. Then, this model is **finetuned** for each individual patient to create a specialized model.

### Step 5: Train the Agnostic Model

**Purpose:** This phase creates a general model that learns the common anatomical features and variations from an entire collection of preprocessed CT scans. This agnostic model serves as a starting point for patient-specific finetuning.

**Script:** `agnostic.py`

**Usage:**
Run the script with the path to the folder of preprocessed CTs (from Step 4) and define an output path for the resulting model.

```bash
python agnostic.py \
    -i /path/to/your/input_data \
    -o /path/to/your/output_directory \
    --name my_cool_experiment
```

-----

### Step 6: Finetune a Patient-Specific Model

**Purpose:** To achieve the highest accuracy for a specific patient, the general agnostic model is specialized using **finetuning**. This step creates a new model that is an expert on a single patient's bone anatomy.

**Script:** `finetuned.py`

**Usage:**
For each patient, run the finetuning script. You must provide the patient's preprocessed CT scan and the path to the agnostic model checkpoint (`.pth` file) created in the previous step. Repeat this process for every patient you intend to register.

```bash
python finetuned.py \
    -i /path/to/your/ct.nii.gz \
    -o /path/to/output/directory_finetuned \
    -c /path/to/your/model_checkpoint.pth \
    --name my_patient_finetuned_model
```
-----

## Registration

### Step 7: Register X-ray to CT with Patient-Specific Model

**Purpose:** This step uses the finetuned, patient-specific model to rapidly and accurately find the 3D pose that aligns the patient's CT scan with a given 2D X-ray. The script outputs the final parameters and trajectory and visualizes the alignment with DRRs.

**Script:** `registration.py`

**Usage:**
Provide the path to the input X-ray, the corresponding patient's preprocessed CT volume, the finetuned model from Step 6, and an output directory.

```bash
python registration.py \
    /path/to/your/xray.png \
    -v /path/to/your/volume.nii.gz \
    -c /path/to/your/model.pth \
    -o /path/to/your/output_folder
```

**Outputs:**

  * `parameters.pt`: A PyTorch file containing the final transform and other registration metrics.
  * `gt.png`: The original input X-ray.
  * `init_image.png`: A DRR generated from the initial guess of the pose.
  * `final_img.png`: A DRR generated from the final, optimized pose.

-----

### Step 8: Create a High-Quality DRR from Final Pose

**Purpose:** The DRR generated during the registration process is optimized for speed. This script allows you to generate a higher-fidelity DRR using the final pose for more precise visual verification or comparison.

**Script:** `create_drr.py`

**Usage:**
Use the CT volume and the `parameters.pt` file generated in the previous step to create a new DRR.

```bash
python create_drr.py \
    --volume /path/to/your/ct_scan.nii.gz \
    --pose_path /path/to/your/parameter_file.pt \
    --output_path /path/to/save/your_drr.png
```

-----

## Evaluation

**Purpose:** The final step is to quantitatively and qualitatively assess the accuracy of the registration. This script generates a single, comprehensive plot that visualizes the results and calculates key similarity metrics between the final DRRs and the ground truth X-ray.

**Script:** `evaluation.py`

**Functionality:** The script loads the ground truth X-ray, the initial pose DRR, and the final DRRs. It then generates a single image file containing:

  * A side-by-side visual comparison of the images.
  * Difference heatmaps showing the per-pixel error between each DRR and the ground truth.
  * Key quantitative metrics, including Mean Squared Error (MSE ↓), Structural Similarity (SSIM ↑), Normalized Mutual Information (NMI ↑), and Normalized Cross-Correlation (NCC ↑).

**Usage:**
Run the script on the results directory generated during the registration and DRR creation steps.

```bash
With saved defaults:

python evaluation.py \
    --data_dir /path/to/patient/results \
    --title_suffix "(p8)"

With specified defaults:

python evaluation.py \
    --data_dir /path/to/patient/results \
    --title_suffix "(p9)" \
    --gt_name ground_truth_xray.png \
    --output_name p9_comparison.png
```
-----

# Utility Scripts

This repository also includes several utility scripts for common data inspection and manipulation tasks.

-----

### Add DICOM Metadata

**Purpose:** X-ray images in DICOM format may sometimes be missing critical metadata. This script allows you to add or update the **Source-to-Detector Distance (SDD)**, which is required for accurate DRR generation.

**Script:** `add_dicom_data.py`

**Usage:**
Provide the input DICOM file, define an output path, and specify the SDD value to be added to the header.

```bash
python add_dicom_data.py \
    --i /path/to/original/xray.dcm \
    --o /path/to/updated/xray_with_sdd.dcm \
    --sdd 1020.0 
```

-----

### Inspect PyTorch Files

**Purpose:** This script is a simple tool for inspecting the contents of PyTorch (`.pt`) files. It's useful for debugging or verifying the keys and tensor shapes within model checkpoints or saved pose files.

**Script:** `read_pt.py`

**Usage:**
Run the script with the path to the `.pt` file you wish to inspect.

```bash
python read_pt.py /path/to/your/file.pt
```

Of course. Here is a section for your `README.md` file describing the visualization scripts.

# Visualization

These scripts help visualize the registration results in both 2D and 3D, which is useful for debugging, analysis, and creating figures.

-----

### Create 3D Visualization

**Purpose**: This script generates a 3D scene that shows the relationship between the CT volume, the X-ray detector, and the camera's final pose. It produces both a static image and an interactive HTML file.

**Script**: `visualize_3d_drr.py`

**Usage**:
Provide the CT volume, the final pose parameters file, and an output directory.

```bash
python visualize_3d_drr.py \
    --volume /path/to/your/volume.nii.gz \
    --parameters /path/to/your/parameters.pt \
    --output_dir /path/to/your/3d_viz_folder
```

-----

### Animate 2D Registration

**Purpose**: This script creates an animated GIF that shows the iterative 2D registration process, visualizing how the DRR aligns with the target X-ray over time.

**Script**: `animate_2d_pose.py`

**Usage**:
Provide the CT volume, the parameters file containing the registration trajectory, and an output path for the GIF.

```bash
python animate_2d_pose.py \
    --volume /path/to/your/ct_scan.nii.gz \
    --parameters /path/to/registration_output/final_pose.pt \
    --output_path /path/to/save/registration_animation.gif
```
