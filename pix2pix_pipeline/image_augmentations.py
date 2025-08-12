import cv2
import os
import albumentations as A
import argparse

'''
Usage:

python image_agumentations.py \
    --input_folder /path/to/your/images \
    --output_folder /path/to/save/augmented_images

'''

def pad_to_square(image, border_color=(0, 0, 0)):
    """Pads an image to a square canvas, placing the image in the center."""
    height, width, _ = image.shape
    
    # Find the longest side
    max_dim = max(height, width)
    
    # Calculate padding
    pad_top = (max_dim - height) // 2
    pad_bottom = max_dim - height - pad_top
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left
    
    # Apply padding
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=border_color)
    return padded_image

def main():
    parser = argparse.ArgumentParser(
        description="Generate affine transformations from paired x-rays and DRR images."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing the images. Naming convention should be a folder for each patient with images gt.png and drr.png."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the output folder where augmented images will be saved."
    )
    args = parser.parse_args()

    aug_gt_dir = os.path.join(args.output_folder, "xrays")
    aug_drr_dir = os.path.join(args.output_folder, "drrs")

    os.makedirs(aug_gt_dir, exist_ok=True)
    os.makedirs(aug_drr_dir, exist_ok=True) 

    transform = A.Compose([
    # These transforms run on the padded image
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.06,
        scale_limit=0.1,
        rotate_limit=20,
        p=0.9
    ),
    A.ElasticTransform(p=0.5)])

    print("Starting augmentation...")
    for root, dirs, files in os.walk(args.input_folder):
        if 'gt.png' in files and 'drr.png' in files:
            folder_name = os.path.basename(root)
            print(f"Processing folder: {folder_name}")

            # Load images
            gt_image = cv2.imread(os.path.join(root, 'gt.png'))
            drr_image = cv2.imread(os.path.join(root, 'drr.png'))
            
            if gt_image is None or drr_image is None: continue

            # Resize DRR to match GT
            h, w, _ = gt_image.shape
            drr_image_resized = cv2.resize(drr_image, (w, h), interpolation=cv2.INTER_LINEAR)

            # Pad both images to a square shape
            gt_padded = pad_to_square(gt_image)
            drr_padded = pad_to_square(drr_image_resized)

            # Inner loop for augmentation
            for i in range(100):
                # Augment the PADDED images
                augmented = transform(image=gt_padded, mask=drr_padded)
                augmented_gt = augmented['image']
                augmented_drr = augmented['mask']

                # Save the final, cropped images
                new_filename = f"{folder_name}_aug_{i}.png"
                cv2.imwrite(os.path.join(aug_gt_dir, new_filename), augmented_gt)
                cv2.imwrite(os.path.join(aug_drr_dir, new_filename), augmented_drr)


    print("Augmentation complete!")


if __name__ == "__main__":
    main()