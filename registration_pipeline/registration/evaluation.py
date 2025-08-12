import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec
import argparse
from pathlib import Path

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import mean_squared_error as mse
from skimage.feature import match_template

'''
Usage:

With saved defaults:

python evaluation.py \
    --data_dir /path/to/patient/results \
    --title_suffix "(p8)"

With specified defaults:

python evaluation.py \
    --data_dir /path/to/patient/results \
    --title_suffix "(p9)" \
    --gt_name ground_truth_xray.png \
    --output_name p9_comparison.png

'''

def load_and_prepare_image(filepath, target_size=None):
    """Loads an image, converts it to grayscale, and returns it as a NumPy array."""
    try:
        img = Image.open(filepath)
        # Resize image if a target size is provided and it doesn't match
        if target_size and img.size != target_size:
            print(f"Resizing {filepath.split('/')[-1]} from {img.size} to {target_size}...")
            # Using LANCZOS for high-quality downsampling/upsampling
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        img = img.convert('L') # Convert to grayscale
        return np.array(img)
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        # Return a placeholder image if the file is not found
        if target_size:
            return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        return np.zeros((256, 256), dtype=np.uint8)

def compare_images(image_paths, output_filename, title_suffix=""):
    """
    Loads images, displays them side-by-side, creates a difference heatmap,
    and calculates quantitative comparison metrics.
    """
    try:
        with Image.open(image_paths['ground_truth']) as img:
            target_pil_size = img.size
    except FileNotFoundError:
        print(f"Error: Ground truth image at {image_paths['ground_truth']} not found. Exiting.")
        return
    
    ground_truth_img = load_and_prepare_image(image_paths['ground_truth'], target_size=target_pil_size)
    initial_drr_img = load_and_prepare_image(image_paths['initial_drr'], target_size=target_pil_size)
    final_xvr_drr_img = load_and_prepare_image(image_paths['final_xvr_drr'], target_size=target_pil_size)
    final_diffpose_drr_img = load_and_prepare_image(image_paths['final_diffpose_drr'], target_size=target_pil_size)

    drr_images = {
        "Initial Pose DRR": initial_drr_img,
        "Final xvr DRR": final_xvr_drr_img,
        "Final diffpose DRR": final_diffpose_drr_img
    }

    # Create the Grid Layout using GridSpec
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.1, wspace=0.1)

    # Dynamic title based on the suffix
    title = f'Comprehensive Registration Evaluation {title_suffix}'.strip()
    fig.suptitle(title, fontsize=20, y=0.97)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(3)]

    # Row 1: Display Images
    axes[0][0].imshow(ground_truth_img, cmap='gray')
    axes[0][0].set_title("Ground Truth X-ray", fontsize=14)
    
    for i, (name, img) in enumerate(drr_images.items()):
        ax = axes[0][i+1]
        ax.imshow(img, cmap='gray')
        ax.set_title(name, fontsize=14)

    # Row 2: Display Difference Heatmaps
    diff_maps = {name: np.abs(ground_truth_img.astype("float") - img.astype("float")) for name, img in drr_images.items()}
    vmax = max(diff.max() for diff in diff_maps.values())
    im = None
    for i, (name, diff) in enumerate(diff_maps.items()):
        ax = axes[1][i+1]
        im = ax.imshow(diff, cmap='viridis', vmin=0, vmax=vmax)
        ax.set_title(f"Difference to Ground Truth", fontsize=12)

    cbar_ax = fig.add_axes([0.92, 0.38, 0.02, 0.25]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Absolute Pixel Difference')

    # Row 3: Display Metrics
    for i, (name, drr_img) in enumerate(drr_images.items()):
        ax = axes[2][i+1]
        if drr_img.size == 0: continue
        
        mse_val = mse(ground_truth_img, drr_img)
        ssim_val = ssim(ground_truth_img, drr_img, data_range=drr_img.max() - drr_img.min())
        nmi_val = nmi(ground_truth_img, drr_img)
        ncc_val = match_template(ground_truth_img, drr_img, pad_input=True).max()

        metrics_text = f"MSE (↓): {mse_val:.2f}\nSSIM (↑): {ssim_val:.4f}\nNMI (↑): {nmi_val:.4f}\nNCC (↑): {ncc_val:.4f}"
        ax.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc='wheat', alpha=0.5))

    for ax_row in axes:
        for ax in ax_row:
            if ax.axison:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)
                    spine.set_visible(True)

    axes[1][0].axis('off')
    axes[2][0].axis('off')
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_filename}")

def main():
    """Parses arguments and runs the image comparison."""
    parser = argparse.ArgumentParser(
        description='Compare registration results against a ground truth X-ray image.'
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="The directory containing the image files."
    )
    parser.add_argument(
        "--gt_name", type=str, default="gt.png",
        help="Filename of the ground truth image. Default: gt.png"
    )
    parser.add_argument(
        "--init_name", type=str, default="init_img.png",
        help="Filename of the initial DRR image. Default: init_img.png"
    )
    parser.add_argument(
        "--xvr_name", type=str, default="final_img.png",
        help="Filename of the final XVR DRR image. Default: final_img.png"
    )
    parser.add_argument(
        "--diffpose_name", type=str, default="drr_image.png",
        help="Filename of the final diffpose DRR image. Default: drr_image.png"
    )
    parser.add_argument(
        "--output_name", type=str, default="comparison_plot.png",
        help="Filename for the output comparison plot. Default: comparison_plot.png"
    )
    parser.add_argument(
        "--title_suffix", type=str, default="",
        help="Optional suffix for the plot title (e.g., a patient ID like 'p8')."
    )
    
    args = parser.parse_args()
    
    image_file_paths = {
        'ground_truth': args.data_dir / args.gt_name,
        'initial_drr': args.data_dir / args.init_name,
        'final_xvr_drr': args.data_dir / args.xvr_name,
        'final_diffpose_drr': args.data_dir / args.diffpose_name
    }
    
    compare_images(image_file_paths, args.output_name, args.title_suffix)

if __name__ == '__main__':
    main()
