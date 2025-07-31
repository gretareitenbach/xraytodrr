import argparse
from pathlib import Path
from subprocess import run

'''
Usage:

python agnostic.py \
    -i /path/to/your/input_data \
    -o /path/to/your/output_directory \
    --name my_cool_experiment

'''

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run xvr train with configurable I/O paths and Wandb name."
    )

    parser.add_argument(
        "-i", "--input_path",
        type=str,
        required=True,
        help="Path to the input training data."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        required=True,
        help="Path to save the output model."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the training run for logging and identification."
    )

    args = parser.parse_args()


    command = f"""
    xvr train \
        -i {Path(args.input_path)} \
        -o {Path(args.output_path)} \
        --r1 -30.0 30.0 \
        --r2 -15.0 15.0 \
        --r3 -25.0 25.0 \
        --tx -150.0 150.0 \
        --ty -800.0 -1400.0 \
        --tz -150.0 150.0 \
        --sdd 1020.0 \
        --height 256 \
        --delx {1/0.943359375} \
        --reverse_x_axis \
        --pretrained \
        --batch_size 8 \
        --n_epochs 1000 \
        --name {args.name} \
        --project xvr
    """
    command = command.strip().split()
    run(command, check=True)

if __name__ == "__main__":
    main()
