import torch
import os
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.utils import save_image
import config
from models.generator import Generator

'''
Usage:

python predict.py \
    --input_path /path/to/your/xray_image_or_directory \
    --output_dir /path/to/save/generated_images \
    --checkpoint /path/to/generator_checkpoint.pth
'''

def preprocess_image(image_path):
    """
    Loads and preprocesses a single image for inference.

    Args:
        image_path (str): The path to the input image.

    Returns:
        Tensor: The preprocessed image tensor.
    """
    image = Image.open(image_path).convert("L")
    transforms = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transforms(image).unsqueeze(0).to(config.DEVICE)

def predict(model, input_tensor, output_path):
    """
    Runs the generator model on an input tensor and saves the output.

    Args:
        model (nn.Module): The trained generator model.
        input_tensor (Tensor): The preprocessed input image tensor.
        output_path (str): The path to save the generated image.
    """
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_tensor = output_tensor * 0.5 + 0.5
        save_image(output_tensor, output_path)
    model.train()
    print(f"Generated image saved to {output_path}")

def main():
    """
    Main function to handle command-line arguments and run prediction.
    """
    parser = argparse.ArgumentParser(description="Generate DRRs from X-ray images using a trained pix2pix model.")
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="Path to a single input X-ray image or a directory of images.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the generated DRR images.")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the generator checkpoint file.")
    
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    gen = Generator(in_channels=config.INPUT_CHANNELS, features=64).to(config.DEVICE)

    print(f"Loading generator checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])

    if os.path.isdir(args.input_path):
        for filename in os.listdir(args.input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input_path, filename)
                input_tensor = preprocess_image(image_path)
                output_path = os.path.join(args.output_dir, filename)
                predict(gen, input_tensor, output_path)
    elif os.path.isfile(args.input_path):
        input_tensor = preprocess_image(args.input_path)
        filename = os.path.basename(args.input_path)
        output_path = os.path.join(args.output_dir, filename)
        predict(gen, input_tensor, output_path)
    else:
        print(f"Error: Input path {args.input_path} is not a valid file or directory.")

if __name__ == "__main__":
    main()
