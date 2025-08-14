import torch
import argparse

from pathlib import Path
from xvr.renderer import initialize_drr
from diffdrr.pose import RigidTransform
from torchvision.utils import save_image

'''
Usage:

python create_drr.py \
    --volume /path/to/your/ct_scan.nii.gz \
    --pose_path /path/to/your/pose_file.pt \
    --output_path /path/to/save/your_drr.png
    
'''

def main():
    """
    Generates and saves a Digitally Reconstructed Radiograph (DRR)
    based on a CT volume and a camera pose file.
    """
    parser = argparse.ArgumentParser(
        description="Generate a DRR from a CT volume and a pose file."
    )
    parser.add_argument(
        "--volume",
        type=Path,
        required=True,
        help="Path to the input CT volume file (e.g., .nii.gz)."
    )
    parser.add_argument(
        "--pose_path",
        type=Path,
        required=True,
        help="Path to the input pose file (e.g., .pt)."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the output DRR image (e.g., drr_image.png)."
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Image dimensions
    HEIGHT = pose["drr"]["height"]
    WIDTH = pose["drr"]["width"]
    DELX = pose["drr"]["delx"]
    DELY = pose["drr"]["dely"]
    
    # Create DRR object
    print(f"Initializing DRR with volume: {args.volume}")
    drr = initialize_drr(
        args.volume,
        mask=None,
        labels=None,
        orientation="PA",
        height=HEIGHT,
        width=WIDTH,
        sdd=1020.0,
        delx=DELX,
        dely=DELY,
        x0=-0.0,
        y0=0.0,
        reverse_x_axis=True,
        renderer="siddon",
        read_kwargs={"bone_attenuation_multiplier": 3.0},
        drr_kwargs={}
    )
    drr.rescale_detector_(0.5)
    print("DRR initialized successfully.")
    
    # Load the final pose
    print(f"Loading pose from: {args.pose_path}")
    final_pose = RigidTransform(torch.load(args.pose_path, weights_only=False, map_location=device)["final_pose"])
    print("Final pose loaded successfully.")
    
    # Generate DRR image
    final_img = drr(final_pose).detach().cpu()
    
    # Save the DRR image
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(final_img, args.output_path, normalize=True)
    print(f"DRR image saved successfully to: {args.output_path}")

if __name__ == "__main__":
    main()

