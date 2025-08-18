# X-ray to DRR Image Translation (pix2pix)

This part of the workflow uses a pix2pix Generative Adversarial Network (GAN) to learn the mapping from 2D X-ray images to Digitally Reconstructed Radiographs (DRRs). The goal is to train a model that can produce a realistic DRR for any given input X-ray.

-----

## The Model

The pix2pix implementation consists of several core components and supporting scripts that work together.

### Core Components

  * **`generator.py`**: This script defines the **Generator** model, which is built using a **U-Net architecture**. Its job is to take an X-ray image as input and generate a corresponding DRR. It uses an encoder-decoder structure with skip connections, allowing it to pass fine-grained details from the input directly to the output, which is crucial for creating sharp, accurate images.

  * **`discriminator.py`**: This script defines the **Discriminator** model, which uses a **PatchGAN** architecture. Instead of classifying the entire image as real or fake, the PatchGAN classifies overlapping patches of the image. This encourages the Generator to produce high-frequency details across the entire image. The Discriminator's role is to act as an adversary, learning to distinguish between real DRRs and the fake ones created by the Generator, thereby pushing the Generator to improve.

### Supporting Scripts

  * **`config.py`**: This is a centralized configuration file that holds all key hyperparameters (like `LEARNING_RATE` and `BATCH_SIZE`), file paths, and model parameters (`IMAGE_SIZE`). This makes it easy to adjust training parameters without modifying the main script code.

  * **`dataset.py`**: This script contains a custom PyTorch `Dataset` class. It's responsible for loading the paired X-ray and DRR images from disk, applying necessary preprocessing steps like resizing and normalization, and serving them to the `DataLoader` during training.

  * **`utils.py`**: A collection of helper functions. Its main roles are saving model checkpoints during training and loading them for inference or continued training. It also includes a function to save a grid of sample images during validation to visually track the Generator's progress.

-----

## The Workflow

The training pipeline is a two-step process: first, the dataset is artificially expanded through augmentation, and then the model is trained on this augmented data.

### Step 1: Augment the Dataset

**Purpose:** To improve the model's ability to generalize and to prevent overfitting, data augmentation is used to artificially expand the training set. This script applies a series of random transformations (like rotations, scaling, and flipping) to the source images, creating a large and varied dataset from a smaller number of initial pairs.

**Script:** `image_augmentations.py`

**Usage:**
Run the script from your terminal, providing the path to your source images and a folder to save the newly created augmented images.

```bash
python image_augmentations.py \
    --input_folder /path/to/your/source_images \
    --output_folder /path/to/save/augmented_images
```

-----

### Step 2: Train the pix2pix Model

**Purpose:** This is the main training step. The script loads the augmented data and trains the Generator and Discriminator models together. The Generator learns to produce realistic DRRs, while the Discriminator learns to distinguish them from real ones. The process is logged to Weights & Biases for real-time monitoring.

**Script:** `train.py`

**Usage:**
Provide the path to your augmented dataset, your Weights & Biases entity, and the desired number of epochs.

```bash
python train.py \
    --input_dir /path/to/augmented/data \
    --wandb_entity your_wandb_username \
    --num_epochs 200
```

-----

### Step 3: Inference (Generate New DRRs)

**Purpose:** After training, this script uses a saved generator checkpoint to perform inference on new, unseen X-ray images. It can process a single image or a directory of images.

**Script:** `predict.py`

**Usage:**
Specify the path to your input X-ray(s), an output directory, and the path to your trained generator model checkpoint.

```bash
python predict.py \
    --input_path /path/to/new/xray.png \
    --output_dir /path/to/save/results \
    --checkpoint checkpoints/gen_epoch_190.pth.tar
```

-----

### Step 4: Evaluate Results

**Purpose:** After creating the predicted GAN-DRRs, this script uses the ground truth registration DRRs, original x-rays, and predicted GAN-DRRs to evaluate the pix2pix model.

**Script:** `evaluate.py`

**Usage:**
Specify the path to your input X-ray(s), GAN-DRRs, ground truth registration-DRRs, and optionally an output directory.

```bash
python evaluate.py \
    --generated_dir results \
    --ground_truth_dir data/testing/drr \
    --input_dir data/testing/xray \
    --output_dir evaluation_results \
```

<img width="657" height="447" alt="comparison_plot" src="https://github.com/user-attachments/assets/f806e0a4-5bd3-4c3c-97ce-9b43b1affaf7" />

