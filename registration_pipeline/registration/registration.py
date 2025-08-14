import argparse
import torch
from pathlib import Path
from subprocess import run
from xvr.renderer import initialize_drr
from diffdrr.pose import RigidTransform
from torchvision.utils import save_image

'''
Usage:

python registration.py \
    /path/to/your/xray.png \
    -v /path/to/your/volume.nii.gz \
    -c /path/to/your/model.pth \
    -o /path/to/your/output_folder \
    --mask /path/to/your/mask.nii.gz \
    --save_paired_drr /path/to/save/final_drr.png

'''

def generate_final_drr(volume_path, pose_path, output_path, device):
    """
    Generates and saves a high-quality DRR from a CT volume and a pose file.
    This function encapsulates the logic from the original create_drr.py script.
    """
    print("\n--- Starting Final DRR Generation ---")

    pose = torch.load(pose_path, weights_only=False, map_location=device)

    # Define DRR parameters
    HEIGHT = pose["drr"]["height"]
    WIDTH = pose["drr"]["width"]
    DELX = pose["drr"]["delx"]
    DELY = pose["drr"]["dely"]
    SDD = 1020.0

    # Initialize the DRR renderer from xvr
    print(f"Initializing DRR with volume: {volume_path}")
    drr = initialize_drr(
        volume_path,
        sdd=SDD,
        height=HEIGHT,
        width=WIDTH,
        delx=DELX,
        dely=DELY,
        x0=-0.0,
        y0=0.0,
        reverse_x_axis=True,
        renderer="siddon",
        read_kwargs={"bone_attenuation_multiplier": 3.0},
    )
    drr.rescale_detector_(0.5)
    drr = drr.to(device)
    print("DRR renderer initialized successfully.")

    # Load the final pose from the registration output
    print(f"Loading final pose from: {pose_path}")
    final_pose_matrix = torch.load(pose_path, map_location=device, weights_only=False)["final_pose"]
    final_pose = RigidTransform(final_pose_matrix)
    print("Final pose loaded successfully.")

    # Generate the DRR image
    with torch.no_grad():
        final_img = drr(final_pose)

    # Save the DRR image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(final_img.cpu(), output_path, normalize=True)
    print(f"✅ Final DRR image saved successfully to: {output_path}")

def main():
    """
    Runs the XVR registration process and optionally creates the final DRR.
    """
    parser = argparse.ArgumentParser(
        description="Register a CT volume to an X-ray and optionally generate the final DRR."
    )

    parser.add_argument(
        "xray",
        type=Path,
        help="Path to the target X-ray or DRR image file."
    )
    parser.add_argument(
        "-v", "--volume",
        type=Path,
        required=True,
        help="Path to the CT volume file (e.g., .nii.gz)."
    )
    parser.add_argument(
        "-c", "--model",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint file (.pth)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the registration results and output images."
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Path to the mask file (optional)."
    )

    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--save_paired_drr",
        type=Path,
        default=None,
        help="If provided, saves the final high-quality DRR to this path after registration."
    )
    # --- END NEW ARGUMENT ---

    args = parser.parse_args()

    # Run the main registration command
    print("--- Starting XVR Registration ---")
    command = f"""
    xvr register model \
        {args.xray} \
        -v {args.volume} \
        -c {args.model} \
        -o {args.output_dir} \
        --scales 12,6 \
        --patience 25 \
        --saveimg \
        --reverse_x_axis \
        --verbose 2 \
        --mask {args.mask} \
        --labels "1" \
    """
    command = command.strip().split()
    run(command, check=True)
    print("--- XVR Registration Complete ---")

    if args.save_paired_drr:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose_file_path = args.output_dir / "parameters.pt"

        if pose_file_path.exists():
            generate_final_drr(
                volume_path=args.volume,
                pose_path=pose_file_path,
                output_path=args.save_paired_drr,
                device=device
            )
        else:
            print(f"ERROR: Could not find pose file at {pose_file_path}. Skipping DRR generation.")

if __name__ == "__main__":
    main()
