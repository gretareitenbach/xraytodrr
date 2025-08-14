import SimpleITK as sitk
from pathlib import Path
import argparse

'''
Usage:

python mirror_xrays.py \
    --input_path /path/to/your/xray_files/ \
    --output_path /path/to/save/mirrored_xrays/
'''

MIRROR_AXIS = 0

def mirror_image(image, axis):
    """
    Mirrors a SimpleITK image along the specified axis.
    
    Args:
        image (sitk.Image): The SimpleITK image to mirror.
        axis (int): The axis (0, 1, or 2) to mirror along.
        
    Returns:
        sitk.Image: The mirrored image.
    """
    # Use the image's dimension to create a transform of the correct size (e.g., 2 for 2D, 3 for 3D)
    image_dim = image.GetDimension()
    mirror_transform = sitk.AffineTransform(image_dim)
    
    # Calculate the center of the image to use as the center of rotation/scaling
    center_index = [(idx - 1) / 2.0 for idx in image.GetSize()]
    center_physical = image.TransformContinuousIndexToPhysicalPoint(center_index)
    mirror_transform.SetCenter(center_physical)

    # Create a scaling vector. -1 along the mirror axis will flip it.
    scale_matrix = [1.0] * image_dim
    scale_matrix[axis] = -1.0
    mirror_transform.Scale(scale_matrix)

    # Use a resampler to apply the transform to the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(mirror_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0) 
    
    return resampler.Execute(image)

def main():
    parser = argparse.ArgumentParser(
        description="Create mirrored images from x-ray scans based on filenames."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to the input directory with x-ray files."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to the output directory."
    )
    
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    # Iterate through all files ending with .nii.gz in the input directory
    for input_path in args.input_path.glob("*.nii.gz"):
        print(f"\nProcessing: {input_path.name}")

        try:
            original_image = sitk.ReadImage(str(input_path))
        except Exception as e:
            print(f"    - ERROR: Could not read file. Skipping. Details: {e}")
            continue
        patient_id = input_path.name.split('_')[0]

        # Check if the filename contains 'right' to decide whether to mirror
        if 'right' in input_path.name.lower():
            print(f"    - 'right' detected. Mirroring image along axis {MIRROR_AXIS}...")
            final_image = mirror_image(original_image, axis=MIRROR_AXIS)
            output_filename = f"{patient_id}_masked_mirrored.nii.gz"
        else:
            print("    - Not a 'right' side image. Passing through without mirroring.")
            final_image = original_image
            output_filename = f"{patient_id}_masked.nii.gz"

        output_path = args.output_path / output_filename
        try:
            sitk.WriteImage(final_image, str(output_path))
            print(f"    - Successfully saved to: {output_path}")
        except Exception as e:
            print(f"    - ERROR: Could not write file. Details: {e}")

if __name__ == "__main__":
    main()