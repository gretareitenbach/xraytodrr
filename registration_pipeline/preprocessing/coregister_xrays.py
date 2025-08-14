import SimpleITK as sitk
import numpy as np
import argparse
from pathlib import Path

'''
Usage:

python coregister_xrays.py \
    --input_dir /path/to/source_xrays \
    --output_dir /path/to/coregistered_xrays \
    --reference_file /path/to/source_xrays/reference_scan.nii.gz

'''

def get_2d_bounding_box_corners(image):
    """Gets the 4 corners of the image's 2D physical bounding box."""
    min_phys = image.GetOrigin()
    max_phys = [o + (s - 1) * sp for o, s, sp in zip(image.GetOrigin(), image.GetSize(), image.GetSpacing())]
    
    corners = [
        (min_phys[0], min_phys[1]), (max_phys[0], min_phys[1]),
        (min_phys[0], max_phys[1]), (max_phys[0], max_phys[1]),
    ]
    return corners

def register_xrays_2d(input_dir, output_dir, reference_filename):
    """
    Runs the 2D-to-2D co-registration process for a directory of X-ray images.
    """
    print(f"--- Starting 2D Co-Registration for Directory: {input_dir.name} ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reference_filename.exists():
        print(f"FATAL: Reference image not found at '{reference_filename}'")
        return

    print(f"Reference (fixed) image: {reference_filename.name}")
    fixed_image = sitk.ReadImage(str(reference_filename), sitk.sitkFloat64)

    for moving_image_path in sorted(input_dir.glob("*.nii.gz")):
        if moving_image_path == reference_filename:
            print(f"Copying reference image {moving_image_path.name} to output directory.")
            sitk.WriteImage(fixed_image, str(output_dir / moving_image_path.name))
            continue

        print(f"\nProcessing (moving) image: {moving_image_path.name}")
        moving_image = sitk.ReadImage(str(moving_image_path), sitk.sitkFloat64)

        # --- REGISTRATION SETUP ---
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares() # MeanSquares is often good for single-modality
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.5
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform)
        
        print("  Running registration...")
        final_transform = registration_method.Execute(fixed_image, moving_image)
        print(f"  Optimizer's stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")

        print("  Calculating output grid to encompass both images...")
        fixed_corners = get_2d_bounding_box_corners(fixed_image)
        moving_corners = get_2d_bounding_box_corners(moving_image)
        transformed_moving_corners = [final_transform.TransformPoint(c) for c in moving_corners]

        all_corners = np.array(fixed_corners + transformed_moving_corners)
        min_coords = all_corners.min(axis=0)
        max_coords = all_corners.max(axis=0)

        # Define the output grid parameters
        output_spacing = fixed_image.GetSpacing()
        output_direction = fixed_image.GetDirection()
        output_origin = min_coords
        output_size = [
            int((max_c - min_c) / spc + 0.5)
            for max_c, min_c, spc in zip(max_coords, min_coords, output_spacing)
        ]

        # --- RESAMPLE AND SAVE ---
        print("  Resampling image onto new grid...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputOrigin(output_origin)
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetOutputDirection(output_direction)
        resampler.SetSize(output_size)
        resampler.SetTransform(final_transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(sitk.sitkFloat64)

        resampled_image = resampler.Execute(moving_image)

        output_path = output_dir / moving_image_path.name
        sitk.WriteImage(resampled_image, str(output_path))
        print(f"  Saved to: {output_path}")

    print(f"\n--- Co-registration for {input_dir.name} complete! ---")

def main():
    """Parses command-line arguments and runs the registration process."""
    parser = argparse.ArgumentParser(
        description="Co-register a directory of 2D X-rays to a reference X-ray."
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing the X-rays (.nii.gz) to be registered."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Output directory to save the co-registered X-rays."
    )
    parser.add_argument(
        "-r", "--reference_file",
        type=Path,
        required=True,
        help="Path to the reference (fixed) X-ray for registration."
    )
    
    args = parser.parse_args()
    
    register_xrays_2d(args.input_dir, args.output_dir, args.reference_file)

if __name__ == "__main__":
    main()