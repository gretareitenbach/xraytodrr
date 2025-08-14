# 2D-3D X-Ray Registration Pipeline

This workflow uses the libraries [XVR](https://github.com/eigenvivek/xvr) and [DiffDRR](https://github.com/eigenvivek/DiffDRR) by Vivek Gopalakrishnan (`eigenvivek`).

## Preprocessing

The goal of preprocessing is to standardize all CT and X-ray data to ensure consistent inputs for the training and registration models.

### Step 1: Prepare CT Scans (Automated)

**Purpose:** This unified script streamlines the entire CT preparation process. It takes a raw CT and its corresponding raw segmentation mask, cleans the mask by keeping only the largest connected components, and then uses the cleaned mask to crop out the specified bones (e.g., femur and tibia). It also automatically handles mirroring for right-sided anatomy.

**Script:** `preprocessing/prepare_ct.py`

**Usage:**
Run the script with the paths to your raw CT and segmentation files. It will create subdirectories for the cropped bones in your specified output directory.

```
python preprocessing/prepare_ct.py \
    --raw_ct_path /path/to/patient/P01_ct.nii.gz \
    --raw_seg_path /path/to/patient/P01_seg.nii.gz \
    --output_dir /path/to/processed_data/P01/
```

-----

### Step 2: Co-Register Cropped CTs

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

### Step 3: Center and Create Isometric Voxels (Manual)

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

## Training

First, a general **agnostic model** is trained on the entire dataset. Then, this model is **finetuned** for each individual patient to create a specialized model.

### Step 4: Train the Agnostic Model

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

### Step 5: Finetune a Patient-Specific Model

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

**Note:** Ensure your 2D X-ray has been prepared according to the steps outlined in the `xray_preprocessing.md` guide before proceeding.

### Step 6: Register X-ray to CT with Patient-Specific Model and Create High-Quality DRR

**Purpose:** This step uses the finetuned, patient-specific model to rapidly and accurately find the 3D pose that aligns the patient's CT scan with a given 2D X-ray. The script outputs the final parameters and trajectory and visualizes the alignment with DRRs. This script also allows you to generate a higher-fidelity DRR using the final pose for more precise visual verification or comparison.

**Script:** `registration.py`

**Usage:**
Provide the path to the input X-ray, the corresponding patient's preprocessed CT volume, the finetuned model from Step 6, and an output directory. Optionally, provide an output directory to save the high quality DRR.

```bash
python registration.py \
    /path/to/your/xray.png \
    -v /path/to/your/volume.nii.gz \
    -c /path/to/your/model.pth \
    -o /path/to/your/output_folder \
    --mask /path/to/your/mask.nii.gz \
    --save_paired_drr /path/to/save/final_drr.png
```

**Outputs:**

  * `parameters.pt`: A PyTorch file containing the final transform and other registration metrics.
  * `gt.png`: The original input X-ray.
  * `init_image.png`: A DRR generated from the initial guess of the pose.
  * `final_img.png`: A DRR generated from the final, optimized pose.

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

<img width="5117" height="4188" alt="comparison_plot" src="https://github.com/user-attachments/assets/16d40e56-8266-4473-93b4-dc9bff8b2317" />
