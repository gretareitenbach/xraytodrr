-----

# Utility Scripts

This repository includes several utility scripts for common data inspection and manipulation tasks.

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
