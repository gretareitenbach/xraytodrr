import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse

'''
Usage:

With saved defaults:

python segmentation_preprocessing.py \
    -i /path/to/your/patientA_seg.nii.gz \
    -o /path/to/your/processed_data

With specified defaults:

python segmentation_preprocessing.py \
    -i /path/to/your/patientB_seg.nii.gz \
    -o /path/to/your/processed_data \
    -l 10 20 30

'''

def get_largest_connected_component(binary_image):
    """
    Takes a binary SimpleITK image (one label) and returns an image
    containing only the largest connected component.
    """
    connected_components = sitk.ConnectedComponentImageFilter().Execute(binary_image)
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabeled_components = relabel_filter.Execute(connected_components)
    largest_component_image = relabeled_components == 1
    return largest_component_image

def main():
    """
    Main function to run the preprocessing pipeline. It takes a multi-label
    segmentation file and keeps only the largest connected component for each
    specified label.
    """

    parser = argparse.ArgumentParser(
        description="Cleans a multi-label segmentation by keeping the largest connected component for each specified label."
    )
    parser.add_argument(
        "-i", "--segmentation_path",
        type=Path,
        required=True,
        help="Path to the input multi-label segmentation file (e.g., .nii.gz)."
    )
    parser.add_argument(
        "-o", "--output_folder_path",
        type=Path,
        required=True,
        help="Path to the folder where the cleaned segmentation will be saved."
    )
    parser.add_argument(
        "-l", "--labels",
        type=int,
        nargs='+',  # This allows for one or more integer arguments
        default=[1, 2, 3, 4],
        help="A space-separated list of integer labels to process. Default: 1 2 3 4"
    )
    
    args = parser.parse_args()
    
    # Create the output directory
    args.output_folder_path.mkdir(parents=True, exist_ok=True)
    
    print("--- Starting Segmentation Preprocessing ---")
    
    if not args.segmentation_path.exists():
        print(f"ERROR: Segmentation file not found at: {args.segmentation_path}")
        return
        
    print(f"Processing Input: {args.segmentation_path.name}")

    try:
        original_seg = sitk.ReadImage(str(args.segmentation_path), sitk.sitkUInt8)
        
        # Create an empty image with the same metadata to store the cleaned results
        final_cleaned_seg = sitk.Image(original_seg.GetSize(), original_seg.GetPixelID())
        final_cleaned_seg.CopyInformation(original_seg)

        for label_id in args.labels:
            print(f"  - Cleaning label {label_id}...")

            binary_mask = original_seg == label_id

            if np.sum(sitk.GetArrayViewFromImage(binary_mask)) == 0:
                print(f"    - Label {label_id} not found. Skipping.")
                continue
            
            largest_component_mask = get_largest_connected_component(binary_mask)
            labeled_component = largest_component_mask * label_id
            final_cleaned_seg = sitk.Add(final_cleaned_seg, sitk.Cast(labeled_component, final_cleaned_seg.GetPixelID()))

        # Save the final reconstructed, cleaned segmentation
        input_stem = args.segmentation_path.stem.replace('.nii', '')
        output_filename = f"{input_stem}_cleaned.nii.gz"
        output_path = args.output_folder_path / output_filename

        sitk.WriteImage(final_cleaned_seg, str(output_path))
        print(f"  -> Saved cleaned segmentation to: {output_path}")

    except Exception as e:
        print(f"  - ERROR processing {args.segmentation_path.name}. Reason: {e}")

    print("\n--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()
