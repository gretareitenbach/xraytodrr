import argparse
from pathlib import Path
from subprocess import run

'''
Usage:

python finetuned.py \
    -i /path/to/your/ct.nii.gz \
    -o /path/to/output/directory_finetuned \
    -c /path/to/your/model_checkpoint.pth \
    --name my_patient_finetuned_model

'''

def main():
    """
    Runs the xvr finetuning process with key parameters provided via
    command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Finetune an XVR model with a specific CT scan."
    )
    parser.add_argument(
        "-i", "--input_ct",
        type=str,
        required=True,
        help="Path to the input CT scan file (e.g., a cropped tibia or femur)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Directory to save the finetuned model and results."
    )
    parser.add_argument(
        "-c", "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint file (.pth) to be finetuned."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="A name for the finetuning Wandb run."
    )
    
    args = parser.parse_args()

    command = f"""
    xvr finetune \
        -i {Path(args.input_ct)} \
        -o {Path(args.output_dir)} \
        -c {Path(args.checkpoint_path)} \
        --name {args.name} \
        --project xvr \
        --batch_size 10 \
        --n_batches_per_epoch 50
    """
    command = command.strip().split()
    run(command, check=True)

if __name__ == "__main__":
    main()
