import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse
import tempfile

'''
Usage: 
python registration_pipeline/preprocessing/prepare_ct.py \
    --raw_ct_path /path/to/patient/P01_ct.nii.gz \
    --raw_seg_path /path/to/patient/P01_seg.nii.gz \
    --output_dir /path/to/processed_data/P01/

'''

def get_largest_connected_component(binary_image):
    """
    Takes a binary SimpleITK image and returns an image containing only the
    largest connected component.
    """
    connected_components = sitk.ConnectedComponentImageFilter().Execute(binary_image)
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabeled_components = relabel_filter.Execute(connected_components)
    largest_component_image = relabeled_components == 1
    return largest_component_image

def clean_segmentation(original_seg, labels_to_process):
    """
    Cleans a multi-label segmentation by keeping only the largest connected
    component for each specified label.
    """
    print("  - Cleaning segmentation mask...")
    final_cleaned_seg = sitk.Image(original_seg.GetSize(), original_seg.GetPixelIDValue())
    final_cleaned_seg.CopyInformation(original_seg)

    for label_id in labels_to_process:
        print(f"    - Processing label {label_id}...")
        binary_mask = original_seg == label_id

        if np.sum(sitk.GetArrayViewFromImage(binary_mask)) == 0:
            print(f"      - Label {label_id} not found. Skipping.")
            continue

        largest_component_mask = get_largest_connected_component(binary_mask)
        # Add the cleaned component back to the final mask, multiplied by its original label ID
        labeled_component = largest_component_mask * label_id
        final_cleaned_seg = sitk.Add(final_cleaned_seg, sitk.Cast(labeled_component, final_cleaned_seg.GetPixelID()))
    
    print("  - Segmentation cleaning complete.")
    return final_cleaned_seg

def get_bounding_box(segmentation_image, label_id):
    """Calculates the bounding box of a specific label in a segmentation image."""
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(segmentation_image)
    if label_id not in label_stats.GetLabels():
        return None
    return label_stats.GetBoundingBox(label_id)

def crop_image_to_bounding_box(image, bounding_box, padding_mm):
    """Crops a SimpleITK image using a bounding box and adds padding."""
    # Convert padding from mm to voxels
    padding_voxels = [int(round(padding_mm / spacing)) for spacing in image.GetSpacing()]
    
    start_index = [bb - pad for bb, pad in zip(bounding_box[:3], padding_voxels)]
    size = [sz + 2 * pad for sz, pad in zip(bounding_box[3:], padding_voxels)]
    
    # Ensure the crop region is within the image boundaries
    image_size = image.GetSize()
    start_index = [max(0, si) for si in start_index]
    end_index = [min(sz, si + s) for sz, si, s in zip(image_size, start_index, size)]
    size = [ei - si for ei, si in zip(end_index, start_index)]
    
    cropper = sitk.RegionOfInterestImageFilter()
    cropper.SetSize(size)
    cropper.SetIndex(start_index)
    return cropper.Execute(image)

def mirror_image(image, axis=0):
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
    # Use the minimum pixel value as the default for the background
    resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))
    return resampler.Execute(image)

def process_bone(ct_image, seg_image, label_id, is_right_sided, output_dir, output_filename, padding_mm):
    """Main processing pipeline for a single bone from a CT scan."""
    print(f"  - Cropping bone for label {label_id} -> '{output_filename}'...")
    bone_mask = seg_image == label_id
    if np.sum(sitk.GetArrayViewFromImage(bone_mask)) == 0:
        print(f"    - WARNING: Label {label_id} not found in cleaned segmentation. Skipping.")
        return

    bbox = get_bounding_box(bone_mask, 1) # Bbox is calculated on the binary mask (label is 1)
    if bbox is None:
        print(f"    - WARNING: Could not calculate bounding box for label {label_id}. Skipping.")
        return

    cropped_ct = crop_image_to_bounding_box(ct_image, bbox, padding_mm)
    
    if is_right_sided:
        print(f"    - Right side detected. Mirroring along axis 0...")
        final_ct = mirror_image(cropped_ct, axis=0)
    else:
        final_ct = cropped_ct
        
    output_path = output_dir / output_filename
    sitk.WriteImage(final_ct, str(output_path))
    print(f"    - Saved cropped CT to: {output_path}")


# --- MAIN SCRIPT LOGIC ---

def main():
    """
    A unified script to preprocess a raw CT scan. It cleans the associated
    segmentation mask and then uses it to crop out specific bones (femur/tibia),
    handling right-sided mirroring automatically.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess a raw CT by cleaning its segmentation and cropping bones."
    )
    parser.add_argument(
        "--raw_ct_path", type=Path, required=True,
        help="Path to the original, full CT scan (.nii.gz)."
    )
    parser.add_argument(
        "--raw_seg_path", type=Path, required=True,
        help="Path to the original, multi-label segmentation mask (.nii.gz)."
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Base directory to save the final cropped CT files."
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

    # Create output directories
    femur_output_dir = args.output_dir / "femur_cts_cropped"
    tibia_output_dir = args.output_dir / "tibia_cts_cropped"
    femur_output_dir.mkdir(parents=True, exist_ok=True)
    tibia_output_dir.mkdir(parents=True, exist_ok=True)

    patient_id = args.raw_ct_path.name.split('_')[0]
    print(f"--- Starting Unified CT Preprocessing for Patient: {patient_id} ---")

    # 1. Load raw images
    print(f"  - Loading CT: {args.raw_ct_path.name}")
    ct_image = sitk.ReadImage(str(args.raw_ct_path), sitk.sitkFloat32)
    
    print(f"  - Loading Segmentation: {args.raw_seg_path.name}")
    original_seg_image = sitk.ReadImage(str(args.raw_seg_path), sitk.sitkUInt8)

    # 2. Resample segmentation to match CT geometry (important alignment step)
    print("  - Aligning segmentation geometry to CT...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(original_seg_image.GetPixelID())
    aligned_seg_image = resampler.Execute(original_seg_image)

    # 3. Clean the aligned segmentation (largest connected component)
    femur_right_label, femur_left_label, tibia_right_label, tibia_left_label = args.labels
    cleaned_seg = clean_segmentation(aligned_seg_image, args.labels)

    # 4. Process each bone using the cleaned segmentation
    print("\nProcessing Femurs...")
    process_bone(ct_image, cleaned_seg, femur_right_label, is_right_sided=True, output_dir=femur_output_dir, output_filename=f"{patient_id}_femur_mirrored.nii.gz", padding_mm=args.padding)
    process_bone(ct_image, cleaned_seg, femur_left_label, is_right_sided=False, output_dir=femur_output_dir, output_filename=f"{patient_id}_femur.nii.gz", padding_mm=args.padding)

    print("\nProcessing Tibias...")
    process_bone(ct_image, cleaned_seg, tibia_right_label, is_right_sided=True, output_dir=tibia_output_dir, output_filename=f"{patient_id}_tibia_mirrored.nii.gz", padding_mm=args.padding)
    process_bone(ct_image, cleaned_seg, tibia_left_label, is_right_sided=False, output_dir=tibia_output_dir, output_filename=f"{patient_id}_tibia.nii.gz", padding_mm=args.padding)

    print(f"\n--- Preprocessing for {patient_id} Complete ---")
    print(f"Final outputs saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
