import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import argparse
from sklearn.model_selection import train_test_split

import config
from dataset import XRayDRRDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils import save_checkpoint, save_some_examples

'''
Usage:

python train.py \ 
    --input_dir /path/to/your/images \
    --wandb_entity your_wandb_entity \
    --num_epochs 200

'''

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, current_epoch):
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch [{current_epoch}/{config.NUM_EPOCHS}]")

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train Discriminator
        with torch.amp.autocast(device_type='cuda'):
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
        
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.amp.autocast(device_type='cuda'):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.LAMBDA_L1
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            wandb.log({
                "Discriminator Loss": D_loss.item(),
                "Generator Loss": G_loss.item(),
                "G_L1_Loss": L1.item(),
                "G_Adversarial_Loss": G_fake_loss.item(),
                "D_Real_Accuracy": torch.sigmoid(D_real).mean().item(),
                "D_Fake_Accuracy": 1 - torch.sigmoid(D_fake).mean().item(),
            })
            
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def validate_fn(gen, loader, l1_loss):
    """Calculates and logs the average L1 loss on the validation set."""
    gen.eval()  # Set generator to evaluation mode
    total_l1_loss = 0.0

    with torch.no_grad():  # No need to calculate gradients for validation
        for x, y in loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            y_fake = gen(x)
            L1 = l1_loss(y_fake, y) * config.LAMBDA_L1
            total_l1_loss += L1.item()

    # Calculate the average L1 loss across all validation batches
    avg_l1_loss = total_l1_loss / len(loader)
    
    # Log the validation metric to Weights & Biases
    wandb.log({"Validation L1 Loss": avg_l1_loss})

    gen.train()  # Set generator back to training mode

def main():
    """
    Main function for a single training run with a standard train/val split.
    """
    parser = argparse.ArgumentParser(description="Generate DRRs from X-ray images using a trained pix2pix model.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Path to a directory of images. Assumes images are in 'xray' and 'drr' subdirectories.")
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        required=True, 
        help="Weights & Biases entity for logging.")
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=200, 
        help="Number of training epochs.")
    
    args = parser.parse_args()
    config.WANDB_ENTITY = args.wandb_entity
    config.NUM_EPOCHS = args.num_epochs
    config.DATA_DIR = args.input_dir

    # Perform a 85/15 train/validation split on the data
    all_files = sorted(os.listdir(os.path.join(config.DATA_DIR, "xray")))
    train_files, val_files = train_test_split(all_files, test_size=0.15, random_state=42)
    
    # Initialize a wandb run
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name="single-model-training-run",
        config={
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "lambda_l1": config.LAMBDA_L1,
            "image_size": config.IMAGE_SIZE,
        }
    )

    # Initialize models and optimizers once
    disc = Discriminator(in_channels=config.INPUT_CHANNELS).to(config.DEVICE)
    gen = Generator(in_channels=config.INPUT_CHANNELS, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    # Setup DataLoaders for the split
    train_dataset = XRayDRRDataset(root_dir=config.DATA_DIR, file_list=train_files)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataset = XRayDRRDataset(root_dir=config.DATA_DIR, file_list=val_files)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Main training loop
    for epoch in range(config.NUM_EPOCHS):

        # Run one epoch of training
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, epoch + 1)

        # Run validation at the end of the epoch
        validate_fn(gen, val_loader, L1_LOSS)

        if config.SAVE_MODEL and epoch % 10 == 0:
            gen_checkpoint_file = f"gen_epoch_{epoch}.pth.tar"
            disc_checkpoint_file = f"disc_epoch_{epoch}.pth.tar"
            save_checkpoint(gen, opt_gen, filename=os.path.join(config.CHECKPOINT_DIR, gen_checkpoint_file))
            save_checkpoint(disc, opt_disc, filename=os.path.join(config.CHECKPOINT_DIR, disc_checkpoint_file))

        save_some_examples(gen, val_loader, epoch, folder=config.SAVED_IMAGES_DIR)

if __name__ == "__main__":
    main()