import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import config

class XRayDRRDataset(Dataset):
    """
    Custom PyTorch Dataset for loading paired X-ray and DRR images.
    """
    def __init__(self, root_dir, file_list):
        """
        Initializes the dataset object.

        Args:
            root_dir (str): The path to the main data directory (e.g., 'data/all_images').
            file_list (list): A list of filenames to be included in this dataset.
        """
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, "xray")
        self.target_dir = os.path.join(root_dir, "drr")
        self.list_files = file_list

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        input_path = os.path.join(self.input_dir, img_file)
        target_path = os.path.join(self.target_dir, img_file)

        input_image = Image.open(input_path).convert("L")
        target_image = Image.open(target_path).convert("L")

        input_image = input_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BICUBIC)
        target_image = target_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BICUBIC)

        input_image = np.array(input_image)
        target_image = np.array(target_image)

        input_image = (input_image / 127.5) - 1.0
        target_image = (target_image / 127.5) - 1.0
        
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
        target_image = np.expand_dims(target_image, axis=0).astype(np.float32)

        input_tensor = torch.from_numpy(input_image)
        target_tensor = torch.from_numpy(target_image)

        return input_tensor, target_tensor