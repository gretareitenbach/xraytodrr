## X-Ray Preprocessing

These steps are required to prepare a raw 2D X-ray for registration. The process involves manual masking and cropping to isolate the anatomy, followed by automated scripts to standardize orientation and add critical metadata.

-----

### Step 1: Manual Masking, Cropping, and Centering

**Purpose:** This initial manual step isolates the specific anatomy of interest within the X-ray. This removes irrelevant background information, reduces noise, and standardizes the image position, which helps the registration model focus on the correct features.

**Script:** *None. This is a manual process in 3D Slicer.*

**Manual Instructions using 3D Slicer:**

1.  **Load X-Ray**: Open the raw X-ray file in 3D Slicer.
2.  **Create Mask**: Go to the `Segment Editor` module and create a new segmentation. Loosely trace the primary anatomical feature (e.g., the femur, a vertebra) to create a basic mask.
3.  **Apply Mask**: Go to the `Mask Volume` module. Select your X-ray as the "Input Volume" and your segmentation as the "Mask". Create a new masked output volume.
4.  **Crop Volume**: Use the `Crop Volume` module to crop the masked volume to the region of interest.
5.  **Center Volume**: In the `Transforms` module, use the **"Center volume"** tool to center the cropped X-ray and harden the transform.
6.  **Save**: Save the resulting centered and cropped X-ray (e.g., as a `.nii.gz` file).

-----

### Step 2: Mirror Right-Sided Images

**Purpose:** To ensure all X-rays have a consistent anatomical orientation for the registration model (e.g., all appear as left-sided anatomy), this script mirrors any images identified as being from the right side.

**Script:** `mirror_xrays.py`

**Usage:**
Provide the path to the directory containing the cropped X-rays from Step 1 and specify an output directory. The script will automatically detect files with "right" in their name and mirror them.

```bash
python mirror_xrays.py \
    --input_path /path/to/cropped_xrays/ \
    --output_path /path/to/mirrored_xrays/
```
-----

### Step 3: Co-Register Cropped X-Rays (Not Tested)

**Purpose:** After cropping, each x-ray scan exists in its own coordinate space. This step aligns, or **co-registers**, all cropped x-ray scans to a common orientation and position. One scan is chosen as a `reference` (or fixed) image, and all other scans are transformed to match it.

**Script:** `coregister_xrays.py`

**Usage:**
Provide the directory of cropped x-rays from Step 2, an output directory, and specify one of the scans to act as the reference.

```bash
python coregister_xrays.py \
    --input_dir /path/to/source_xrays \
    --output_dir /path/to/coregistered_xrays \
    --reference_file /path/to/source_xrays/reference_scan.nii.gz
```

-----

### Step 4: Convert to DICOM Format

**Purpose:** The final steps of preprocessing require the image to be in DICOM format to store essential metadata in the file's header.

**Script:** *None. This is a manual process in 3D Slicer.*

**Manual Instructions using 3D Slicer:**

1.  **Load Data**: Open the processed X-ray from Step 2 (e.g., `patient_masked.nii.gz` or `patient_masked_mirrored.nii.gz`).
2.  **Save as DICOM**: Go to **File \> Save**. In the save dialog, select "DICOM" as the file format and save the image.

-----

### Step 5: Add DICOM Metadata (SDD)

**Purpose:** A crucial piece of metadata for accurate registration is the **Source-to-Detector Distance (SDD)**, which defines the geometry of the X-ray machine. This script embeds a specified SDD value directly into the DICOM file header.

**Script:** `resources/add_dicom_data.py`

**Usage:**
Provide the input DICOM from Step 3, define an output path for the new file, and specify the SDD value.

```bash
python add_dicom_data.py \
    --i /path/to/original/xray.dcm \
    --o /path/to/final_preprocessed_xray.dcm \
    --sdd 1020.0 
```

-----

The final, preprocessed DICOM file from this section is now ready to be used as the X-ray input for **Step 7: Register X-ray to CT with Patient-Specific Model** in the registration pipeline.
