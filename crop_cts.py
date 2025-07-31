import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse

'''
Usage:

With saved defaults:

python crop_cts.py \
    --base_dir /path/to/patient/data \
    --ct_filename P01_ct.nii.gz \
    --seg_filename segmentations/P01_seg.nii.gz

With specified defaults:

python crop_cts.py \
    --base_dir /path/to/patient/data \
    --ct_filename P01_ct.nii.gz \
    --seg_filename segmentations/P01_seg.nii.gz \
    --labels 10 11 20 21 \
    --padding 10.0

'''

# Constants
MIRROR_AXIS = 0
FEMUR_OUTPUT_FOLDER = "femur_cts_cropped" # will be created inside the base directory
TIBIA_OUTPUT_FOLDER = "tibia_cts_cropped" # will be created inside the base directory

def get_bounding_box(segmentation_image, label_id):
    """Calculates the bounding box of a specific label in a segmentation image."""
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(segmentation_image)
    if label_id not in label_stats.GetLabels():
        return None
    return label_stats.GetBoundingBox(label_id)

def crop_image_to_bounding_box(image, bounding_box, padding_mm):
    """Crops a SimpleITK image using a bounding box and adds padding."""
    padding_voxels = [int(round(padding_mm / spacing)) for spacing in image.GetSpacing()]
    start_index = [bb - pad for bb, pad in zip(bounding_box[:3], padding_voxels)]
    size = [sz + 2 * pad for sz, pad in zip(bounding_box[3:], padding_voxels)]
    image_size = image.GetSize()
    start_index = [max(0, si) for si in start_index]
    end_index = [min(sz, si + s) for sz, si, s in zip(image_size, start_index, size)]
    size = [ei - si for ei, si in zip(end_index, start_index)]
    cropper = sitk.RegionOfInterestImageFilter()
    cropper.SetSize(size)
    cropper.SetIndex(start_index)
    return cropper.Execute(image)

def mirror_image(image, axis):
    """Mirrors a SimpleITK image along the specified axis."""
    mirror_transform = sitk.AffineTransform(3)
    center = image.TransformContinuousIndexToPhysicalPoint([(idx - 1) / 2.0 for idx in image.GetSize()])
    mirror_transform.SetCenter(center)
    scale_matrix = [1.0, 1.0, 1.0]
    scale_matrix[axis] = -1.0
    mirror_transform.Scale(scale_matrix)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(mirror_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(image.GetPixelIDValue()))
    return resampler.Execute(image)

def process_bone(ct_image, seg_image, label_id, is_right_sided, output_dir, output_filename, padding_mm):
    """Main processing pipeline for a single bone."""
    print(f"  - Processing label {label_id} for output '{output_filename}'...")
    bone_mask = seg_image == label_id
    if np.sum(sitk.GetArrayViewFromImage(bone_mask)) == 0:
        print(f"    - WARNING: Label {label_id} not found in segmentation. Skipping.")
        return
    bbox = get_bounding_box(bone_mask, 1)
    if bbox is None:
        print(f"    - WARNING: Could not calculate bounding box for label {label_id}. Skipping.")
        return
    cropped_ct = crop_image_to_bounding_box(ct_image, bbox, padding_mm)
    if is_right_sided:
        print(f"    - Right side detected. Mirroring along axis {MIRROR_AXIS}...")
        final_ct = mirror_image(cropped_ct, axis=MIRROR_AXIS)
    else:
        final_ct = cropped_ct
    output_path = output_dir / output_filename
    sitk.WriteImage(final_ct, str(output_path))
    print(f"    - Saved to: {output_path}")

def main():
    """Main function to run the cropping pipeline from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Crops femur and tibia bones from a CT scan using a segmentation mask."
    )
    parser.add_argument(
        "--base_dir", type=Path, required=True, 
        help="Base directory containing the input files and for writing output folders."
    )
    parser.add_argument(
        "--ct_filename", type=str, required=True, 
        help="Filename of the CT scan (e.g., 'patient1_ct.nii.gz')."
    )
    parser.add_argument(
        "--seg_filename", type=str, required=True, 
        help="Filename of the segmentation mask, can be in a subdirectory (e.g., 'segs/patient1_seg.nii.gz')."
    )
    parser.add_argument(
        "--labels", type=int, nargs=4, default=[3, 4, 1, 2],
        metavar=("FEMUR_R", "FEMUR_L", "TIBIA_R", "TIBIA_L"),
        help="A list of 4 integer labels for [Right Femur, Left Femur, Right Tibia, Left Tibia]. Default: 3 4 1 2"
    )
    parser.add_argument(
        "--padding", type=float, default=5.0,
        help="Padding in millimeters to add around the cropped bone. Default: 5.0"
    )
    
    args = parser.parse_args()

    # Extract labels from the parsed arguments
    femur_right_label, femur_left_label, tibia_right_label, tibia_left_label = args.labels
    
    # Infer patient ID from the CT filename for consistent output naming
    patient_id = Path(args.ct_filename).name.split('_')[0]
    
    ct_path = args.base_dir / args.ct_filename
    seg_path = args.base_dir / args.seg_filename
    
    femur_output_dir = args.base_dir / FEMUR_OUTPUT_FOLDER
    tibia_output_dir = args.base_dir / TIBIA_OUTPUT_FOLDER
    femur_output_dir.mkdir(parents=True, exist_ok=True)
    tibia_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting CT Cropping for Patient ID: {patient_id} ---")

    if not ct_path.exists() or not seg_path.exists():
        print(f"FATAL ERROR: Input file not found. Check paths.")
        print(f"  - CT path checked: {ct_path}")
        print(f"  - Seg path checked: {seg_path}")
        return
        
    print(f"  - Loading CT: {ct_path.name}")
    print(f"  - Loading Segmentation: {seg_path.name}")
    ct_image = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    original_seg_image = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)

    print("  - Resampling segmentation to align with CT geometry...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(original_seg_image.GetPixelID())
    seg_image = resampler.Execute(original_seg_image)

    # Process Femurs
    print("\nProcessing Femurs...")
    process_bone(ct_image, seg_image, femur_right_label, is_right_sided=True, output_dir=femur_output_dir, output_filename=f"{patient_id}_femur_mirrored.nii.gz", padding_mm=args.padding)
    process_bone(ct_image, seg_image, femur_left_label, is_right_sided=False, output_dir=femur_output_dir, output_filename=f"{patient_id}_femur.nii.gz", padding_mm=args.padding)

    # Process Tibias
    print("\nProcessing Tibias...")
    process_bone(ct_image, seg_image, tibia_right_label, is_right_sided=True, output_dir=tibia_output_dir, output_filename=f"{patient_id}_tibia_mirrored.nii.gz", padding_mm=args.padding)
    process_bone(ct_image, seg_image, tibia_left_label, is_right_sided=False, output_dir=tibia_output_dir, output_filename=f"{patient_id}_tibia.nii.gz", padding_mm=args.padding)

    print("\n--- Preprocessing Complete ---")
    print(f"Processed femurs saved to: {femur_output_dir}")
    print(f"Processed tibias saved to: {tibia_output_dir}")

if __name__ == "__main__":
    main()