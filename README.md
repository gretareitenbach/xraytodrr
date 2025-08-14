# X-Ray to DRR: A Deep Learning Pipeline for 2D/3D Medical Image Registration and Translation

This repository contains a comprehensive, two-part deep learning pipeline designed to solve a critical challenge in medical imaging: the alignment of 2D X-ray images with 3D CT scans.

1.  **Registration Pipeline:** This first part uses a powerful, gradient-based registration framework (**XVR** and **DiffDRR**) to accurately determine the 3D pose of a CT scan that matches a 2D X-ray. The key output of this process is a perfectly aligned pair of images: the original X-ray and a synthetic X-ray (a Digitally Reconstructed Radiograph, or DRR) generated from the CT at the correct pose.

2.  **Translation Pipeline:** The paired (X-ray, DRR) images created by the registration pipeline serve as the ground-truth training data for a **pix2pix GAN**. This second pipeline learns the complex mapping from the domain of real X-rays to the domain of synthetic DRRs. The ultimate goal is to train a generator model that can take *any* new X-ray and produce a corresponding, high-quality DRR without needing the original CT scan.

-----

## 🚀 Key Features

  * **Automated 2D/3D Registration:** A robust pipeline to align CT volumes with X-ray images.
  * **Paired Dataset Creation:** The registration pipeline's primary function is to generate the aligned (X-ray, DRR) image pairs needed for the translation task.
  * **Image-to-Image Translation:** A pix2pix GAN that learns the stylistic and anatomical mapping from X-rays to DRRs.
  * **Modular & Script-Based:** The entire workflow is broken down into manageable, command-line-driven scripts for preprocessing, training, and inference.
  * **Comprehensive Tooling:** Includes scripts for data preprocessing, visualization, and quantitative evaluation.

-----

## 📂 Directory Structure

```
.
├── registration_pipeline/ # All scripts for 2D/3D registration
│   ├── preprocessing/
│   ├── registration/
│   ├── training/
│   └── visualization/
│
├── translation_pipeline/      # All scripts for image-to-image translation
│   ├── models/
│   ├── image_augmentations.py
│   ├── train.py
│   └── predict.py
│
├── environment.yml        # Conda environment for dependency management
└── README.md              # This file
```

-----

## ⚙️ Getting Started

### Prerequisites

  * Python 3.10
  * [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management
  * A CUDA-enabled GPU is highly recommended for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/xraytodrr.git
    cd xraytodrr
    ```
2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate xray_to_drr_env
    ```

### Data Preparation

This pipeline requires your own 3D CT scans (in `.nii.gz` format) and 2D X-ray images (e.g., `.dcm`, `.png`). You will need to organize your data before running the scripts. We recommend creating a `data/` directory at the root of the project to store your raw and processed files.

-----

## 📋 How to Run the Pipeline

This workflow is divided into two main parts. You must complete Part 1 to generate the necessary data for Part 2.

### Part 1: Generate Paired Data with the Registration Pipeline

The goal of this part is to produce a dataset of perfectly aligned (X-ray, DRR) image pairs.

1.  **Preprocess CTs and X-Rays:** Follow the detailed steps in the `registration_pipeline/README.md` and `registration_pipeline/preprocessing/xray_preprocessing.md` to clean, crop, and co-register your raw medical images. This is a critical step to ensure data consistency.
2.  **Train Registration Models:** Use the scripts in `registration_pipeline/training/` to train an agnostic model on your entire CT dataset and then finetune patient-specific models. See the `registration_pipeline/README.md` for detailed instructions.
3.  **Run Registration:** For each patient, use the finetuned model to register their CT to a corresponding X-ray. This will output the final 3D pose in a `parameters.pt` file.
    ```bash
    python registration_pipeline/registration/registration.py \
        /path/to/your/xray.dcm \
        -v /path/to/your/volume.nii.gz \
        -c /path/to/your/model.pth \
        -o /path/to/your/output_folder
    ```
4.  **Create High-Quality DRR:** Use the `parameters.pt` file from the previous step to generate a high-fidelity DRR. This DRR is now perfectly aligned with the input X-ray.
    ```bash
    python registration_pipeline/registration/create_drr.py \
        --volume /path/to/your/ct_scan.nii.gz \
        --pose_path /path/to/your/output_folder/parameters.pt \
        --output_path /path/to/paired_data/drrs/patient_01.png
    ```
    Your original X-ray (e.g., `gt.png`) and this newly created DRR (`patient_01.png`) form one paired sample for the next stage. Repeat this for all patients to build your dataset.

### Part 2: Train the X-ray to DRR Translation Model

Now, we use the paired dataset to train the pix2pix GAN.

1.  **Organize Paired Data:** Create a directory structure for your paired images. The `pix2pix_pipeline/dataset.py` script expects images to be in `xray` and `drr` subfolders with matching filenames.
    ```
    paired_data/
    ├── xray/
    │   ├── patient_01.png
    │   └── patient_02.png
    └── drr/
        ├── patient_01.png
        └── patient_02.png
    ```
2.  **Augment the Dataset (Optional but Recommended):** To improve model robustness, create a larger, more diverse dataset by applying random transformations.
    ```bash
    python pix2pix_pipeline/image_augmentations.py \
        --input_folder /path/to/paired_data/ \
        --output_folder /path/to/augmented_data/
    ```
3.  **Train the pix2pix Model:** Run the main training script on the augmented data. Monitor the training progress using Weights & Biases.
    ```bash
    python pix2pix_pipeline/train.py \
        --input_dir /path/to/augmented_data/ \
        --wandb_entity your_wandb_username \
        --num_epochs 200
    ```
4.  **Perform Inference:** Once training is complete, use the saved generator model to translate a new, unseen X-ray into a DRR.
    ```bash
    python pix2pix_pipeline/predict.py \
        --input_path /path/to/new/xray.png \
        --output_dir /path/to/save/results \
        --checkpoint pix2pix_pipeline/checkpoints/gen_epoch_190.pth.tar
    ```

-----

## 🙏 Acknowledgements

This work is built upon the excellent research and open-source libraries created by Vivek Gopalakrishnan.

  * [**XVR**](https://github.com/eigenvivek/xvr)
  * [**DiffDRR**](https://github.com/eigenvivek/DiffDRR)
