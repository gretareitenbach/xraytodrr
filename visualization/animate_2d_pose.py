import torch
import argparse
from pathlib import Path
from xvr.renderer import initialize_drr
from diffdrr.visualization import animate

'''
Usage:

python animate_2d_pose.py \
    --volume /path/to/your/ct_scan.nii.gz \
    --parameters /path/to/registration_output/final_pose.pt \
    --output_path /path/to/save/registration_animation.gif
    
'''

def main():
    """
    Creates an animated GIF of a 2D registration process from a volume
    and a trajectory data file.
    """
    parser = argparse.ArgumentParser(
        description="Animate a 2D registration trajectory from a parameters file."
    )
    parser.add_argument(
        "-v", "--volume",
        type=str,
        required=True,
        help="Path to the input CT volume file (e.g., .nii.gz)."
    )
    parser.add_argument(
        "-p", "--parameters",
        type=str,
        required=True,
        help="Path to the .pt file containing the registration trajectory."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        required=True,
        help="Path to save the output animation (e.g., animation.gif)."
    )
    
    args = parser.parse_args()
    
    device = torch.device("cpu")
    
    # Constants for DRR generation
    HEIGHT = 3362
    WIDTH = 1038
    DELX = 0.148
    DELY = 0.148
    
    # Create DRR object
    print(f"Initializing DRR with volume: {args.volume}")
    drr = initialize_drr(Path(args.volume),
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
                drr_kwargs={})
    drr.rescale_detector_(0.5)
    drr = drr.to(device)
    print("DRR initialized successfully.")
    
    # Get trajectory dataframe from the parameters file
    print(f"Loading trajectory from: {args.parameters}")
    df = torch.load(args.parameters, weights_only=False, map_location=device)["trajectory"].iloc[:, :7]
    df.columns = ['alpha', 'beta', 'gamma', 'bx', 'by', 'bz', 'loss']
    print("Trajectory dataframe loaded successfully.")
    
    # Ensure the output directory exists
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Animate the registration
    print(f"Generating animation, this may take a moment...")
    animate(
        out=Path(args.output_path),
        parameterization="euler_angles",
        convention="XYZ",
        drr=drr,
        df=df
    )
    print(f"Animation saved successfully to: {args.output_path}")

if __name__ == "__main__":
    main()
