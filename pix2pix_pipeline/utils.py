import torch
import torchvision
import os
import config
import wandb

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Saves the model and optimizer states to a file.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        filename (str): The path to save the checkpoint file.
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Loads a model and optimizer state from a checkpoint file.

    Args:
        checkpoint_file (str): The path to the checkpoint file.
        model (nn.Module): The model to load the state into.
        optimizer (Optimizer): The optimizer to load the state into.
        lr (float): The learning rate to set for the optimizer.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_some_examples(gen, val_loader, epoch, folder):
    """
    Saves a grid of images to see the generator's performance.
    It saves a triplet: input_image, generated_image, real_target_image.

    Args:
        gen (nn.Module): The generator model.
        val_loader (DataLoader): The DataLoader for the validation set.
        epoch (int): The current epoch number, used for naming the file.
        folder (str): The directory where the image grid will be saved.
    """
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        
        image_grid = torchvision.utils.make_grid(torch.cat((x, y_fake, y), 0))
        save_path = os.path.join(folder, f"sample_epoch_{epoch}.png")
        os.makedirs(folder, exist_ok=True)
        torchvision.utils.save_image(image_grid, save_path)
        
        wandb.log({f"Validation Samples Epoch {epoch}": wandb.Image(image_grid)})
        
    gen.train()

