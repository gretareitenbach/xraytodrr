import argparse
from pathlib import Path
from subprocess import run

'''
Usage:

python registration.py \
    /path/to/your/xray.png \
    -v /path/to/your/volume.nii.gz \
    -c /path/to/your/model.pth \
    -o /path/to/your/output_folder \
    --mask /path/to/your/mask.nii.gz \

'''

def main():
    """
    Runs the XVR registration process using command-line arguments for file paths.
    """
    parser = argparse.ArgumentParser(
        description="Register a CT volume to an X-ray using a trained XVR model."
    )

    parser.add_argument(
        "xray",
        type=str,
        help="Path to the target X-ray or DRR image file."
    )

    parser.add_argument(
        "-v", "--volume",
        type=str,
        required=True,
        help="Path to the CT volume file (e.g., .nii.gz)."
    )
    parser.add_argument(
        "-c", "--model",
        type=str,
        required=True,
        help="Path to the trained model checkpoint file (.pth)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Directory to save the registration results and output images."
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to the mask file (optional)."
    )
    
    args = parser.parse_args()

    command = f"""
    xvr register model \
        {Path(args.xray)} \
        -v {Path(args.volume)} \
        -c {Path(args.model)} \
        -o {Path(args.output_dir)} \
        --scales 12,6 \
        --patience 25 \
        --saveimg \
        --reverse_x_axis \
        --verbose 2 \
        --mask {Path(args.mask)} \
        --labels "1" \
    """
    command = command.strip().split()
    run(command, check=True)

if __name__ == "__main__":
    main()
