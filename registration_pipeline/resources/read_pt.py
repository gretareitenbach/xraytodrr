import torch
import argparse
from pathlib import Path

def main():
    """
    Loads a PyTorch .pt file and prints its contents.
    """
    parser = argparse.ArgumentParser(
        description="Reads a PyTorch .pt file and prints the keys and values."
    )
    
    # Add a positional argument for the file path
    parser.add_argument(
        "file_path",
        type=Path,
        help="The full path to the .pt file to be read."
    )
    
    args = parser.parse_args()
    
    try:
        # Load data from the specified file
        data = torch.load(args.file_path, weights_only=False)
        
        # Check if the loaded data is a dictionary (like a model state_dict)
        if isinstance(data, dict):
            print(f"--- Contents of {args.file_path.name} ---")
            for name, param in data.items():
                # Print key and the tensor's shape and device
                print(f"Key: {name}")
                if isinstance(param, torch.Tensor):
                    print(f"  └─ Tensor(size={param.size()}, device={param.device}, dtype={param.dtype})")
                else:
                    print(f"  └─ Value: {param}")
                print("-" * 20)
        else:
            # If it's not a dictionary, print the whole object
            print(data)

    except FileNotFoundError:
        print(f"Error: File not found at {args.file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()