import SimpleITK as sitk
import numpy as np
import argparse
from pathlib import Path

'''
Usage:

python coregister_bones.py \
    --input_dir /path/to/source_bones \
    --output_dir /path/to/coregistered_bones \
    --reference_file /path/to/source_bones/reference_scan.nii.gz

'''

def get_physical_bounding_box_corners(image):
    """Gets the 8 corners of the image's physical bounding box."""
    min_phys = image.GetOrigin()
    max_phys = [o + (s - 1) * sp for o, s, sp in zip(image.GetOrigin(), image.GetSize(), image.GetSpacing())]
    
    corners = [
        (min_phys[0], min_phys[1], min_phys[2]), (max_phys[0], min_phys[1], min_phys[2]),
        (min_phys[0], max_phys[1], min_phys[2]), (min_phys[0], min_phys[1], max_phys[2]),
        (max_phys[0], max_phys[1], min_phys[2]), (max_phys[0], min_phys[1], max_phys[2]),
        (min_phys[0], max_phys[1], max_phys[2]), (max_phys[0], max_phys[1], max_phys[2]),
    ]
    return corners


def register_directory(input_dir, output_dir, reference_filename):
    """
    Runs the 3D-to-3D co-registration process for a directory of images.
    """
    print(f"--- Starting Co-Registration for Directory: {input_dir.name} ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the fixed reference image
    if not reference_filename.exists():
        print(f"FATAL: Reference image not found at '{reference_filename}'")
        return

    print(f"Reference (fixed) image: {reference_filename.name}")
    fixed_image = sitk.ReadImage(str(reference_filename), sitk.sitkFloat64)

    # Loop through all files
    for moving_image_path in sorted(input_dir.glob("*.nii.gz")):
        if moving_image_path == reference_filename:
            print(f"Copying reference image {moving_image_path.name} to output directory.")
            sitk.WriteImage(fixed_image, str(output_dir / moving_image_path.name))
            continue

        print(f"\nProcessing (moving) image: {moving_image_path.name}")
        moving_image = sitk.ReadImage(str(moving_image_path), sitk.sitkFloat64)

        # Registration
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.02)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.5
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform)
        
        print("  Running registration...")
        final_transform = registration_method.Execute(fixed_image, moving_image)
        print(f"  Optimizer's stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")

        # 4. COMPUTE AN OUTPUT GRID THAT AVOIDS CROPPING
        print("  Calculating output grid to encompass both images...")
        
        fixed_corners = get_physical_bounding_box_corners(fixed_image)
        moving_corners = get_physical_bounding_box_corners(moving_image)
        transformed_moving_corners = [final_transform.TransformPoint(c) for c in moving_corners]

        all_corners = np.array(fixed_corners + transformed_moving_corners)
        min_coords = all_corners.min(axis=0)
        max_coords = all_corners.max(axis=0)

        output_spacing = fixed_image.GetSpacing()
        output_direction = fixed_image.GetDirection()
        output_origin = min_coords
        output_size = [
            int((max_c - min_c) / spc + 1.5) # Add 0.5 for rounding, 1 for full extent
            for max_c, min_c, spc in zip(max_coords, min_coords, output_spacing)
        ]

        # Resample and save
        print("  Resampling image onto new grid...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputOrigin(output_origin)
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetOutputDirection(output_direction)
        resampler.SetSize(output_size)
        resampler.SetTransform(final_transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000) # Background value for CT
        resampler.SetOutputPixelType(sitk.sitkFloat64)

        resampled_image = resampler.Execute(moving_image)

        output_path = output_dir / moving_image_path.name
        sitk.WriteImage(resampled_image, str(output_path))
        print(f"  Saved to: {output_path}")

    print(f"\n--- Co-registration for {input_dir.name} complete! ---")

def main():
    """Parses command-line arguments and runs the registration process."""
    parser = argparse.ArgumentParser(
        description="Co-register a directory of CT scans to a reference CT scan."
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing the CT or Xray scans (.nii.gz) to be registered."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Output directory to save the co-registered CT or Xray scans."
    )
    parser.add_argument(
        "-r", "--reference_file",
        type=Path,
        required=True,
        help="Path to the reference (fixed) CT or Xray scan for registration."
    )
    
    args = parser.parse_args()
    
    register_directory(args.input_dir, args.output_dir, args.reference_file)

if __name__ == "__main__":
    main()
