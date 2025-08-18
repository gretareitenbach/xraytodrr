import os
import argparse
import random
import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

'''
Usage:
python evaluate.py \
    --generated_dir results \
    --ground_truth_dir data/testing/drr \
    --input_dir data/testing/xray \
    --output_dir evaluation_results \
    --output_filename final_comparison.png
'''

def calculate_metrics(generated_path, ground_truth_path):
    """
    Calculates comparison metrics between a generated and ground truth image.

    Args:
        generated_path (str): Filepath to the generated image.
        ground_truth_path (str): Filepath to the ground truth image.

    Returns:
        tuple: A tuple containing MAE, SSIM, and NCC scores.
    """
    # Load images and resize ground truth to match the generated image's size
    generated_img_pil = Image.open(generated_path).convert("L")
    ground_truth_img_pil = Image.open(ground_truth_path).convert("L")
    target_size = generated_img_pil.size
    ground_truth_img_pil = ground_truth_img_pil.resize(target_size, Image.Resampling.LANCZOS)

    # Convert images to NumPy arrays for calculations
    generated_img = np.array(generated_img_pil)
    ground_truth_img = np.array(ground_truth_img_pil)
    
    # Calculate metrics
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(generated_img.astype(np.float32) - ground_truth_img.astype(np.float32)))
    # Structural Similarity Index (SSIM)
    ssim_score = ssim(ground_truth_img, generated_img, data_range=255)
    # Normalized Cross-Correlation (NCC)
    ncc_score = match_template(ground_truth_img, generated_img, pad_input=True).max()
    
    return mae, ssim_score, ncc_score

def main():
    """
    Main function to run the evaluation, calculate metrics, and generate the comparison plot.
    """
    parser = argparse.ArgumentParser(description="Evaluate generated images and create a consolidated visualization.")
    parser.add_argument(
        "--generated_dir", 
        required=True, 
        help="Directory of generated images.")
    parser.add_argument(
        "--ground_truth_dir", 
        required=True, 
        help="Directory of ground truth images.")
    parser.add_argument(
        "--input_dir", 
        required=True, 
        help="Directory of original input images.")
    parser.add_argument(
        "--output_dir", 
        default="evaluation_results", 
        help="Directory to save the final plot.")
    parser.add_argument(
        "--output_filename", 
        default="comparison_plot.png", 
        help="Filename for the final plot.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_mae, all_ssim, all_ncc = [], [], []
    image_files = sorted(os.listdir(args.generated_dir))
    
    num_examples = 3
    if len(image_files) < num_examples:
        print(f"Warning: Found only {len(image_files)} images, visualizing all of them.")
        num_examples = len(image_files)
        
    examples_to_visualize = random.sample(image_files, k=num_examples)
    plot_data = []

    print("Calculating metrics and collecting data for visualization...")
    for filename in tqdm(image_files):
        try:
            generated_path = os.path.join(args.generated_dir, filename)
            ground_truth_path = os.path.join(args.ground_truth_dir, filename)
            input_path = os.path.join(args.input_dir, filename)

            if os.path.exists(ground_truth_path) and os.path.exists(input_path):
                mae, ssim_score, ncc_score = calculate_metrics(generated_path, ground_truth_path)
                all_mae.append(mae)
                all_ssim.append(ssim_score)
                all_ncc.append(ncc_score)

                # If the file is one of our chosen examples, collect its data for plotting
                if filename in examples_to_visualize:
                    plot_data.append({
                        "input": Image.open(input_path).convert("L"),
                        "generated": Image.open(generated_path).convert("L"),
                        "ground_truth": Image.open(ground_truth_path).convert("L"),
                        "metrics": {"mae": mae, "ssim": ssim_score, "ncc": ncc_score}
                    })
        except (OSError, UnidentifiedImageError) as e:
            print(f"\nWarning: Skipping corrupted or truncated image file: {filename}. Error: {e}")
            continue

    if plot_data:
        print(f"Generating consolidated plot: {args.output_filename}")
        
        num_rows = len(plot_data)
        fig = plt.figure(figsize=(22, 5 * num_rows))
        gs = gridspec.GridSpec(num_rows, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.7])
        
        column_titles = ["Real X-ray", "Ground-Truth DRR", "GAN-DRR", "Absolute Error", "Metrics"]

        error_maps = []
        for data in plot_data:
            target_size = data["generated"].size
            gt_resized = data["ground_truth"].resize(target_size, Image.Resampling.LANCZOS)
            error_map = np.abs(np.array(data["generated"], dtype=float) - np.array(gt_resized, dtype=float))
            error_maps.append(error_map)

        global_vmax = np.percentile(np.array(error_maps), 99) if error_maps else 1.0

        for i, data in enumerate(plot_data):
            target_size = data["generated"].size
            input_img = data["input"].resize(target_size, Image.Resampling.LANCZOS)
            ground_truth_img = data["ground_truth"].resize(target_size, Image.Resampling.LANCZOS)

            axes = [fig.add_subplot(gs[i, j]) for j in range(5)]
            
            axes[0].imshow(input_img, cmap='gray')
            axes[1].imshow(ground_truth_img, cmap='gray')
            axes[2].imshow(data["generated"], cmap='gray')
            
            im = axes[3].imshow(error_maps[i], cmap='jet', vmin=0, vmax=global_vmax)

            metrics = data["metrics"]
            metrics_text = (f"MAE (↓): {metrics['mae']:.2f}\n"
                            f"SSIM (↑): {metrics['ssim']:.4f}\n"
                            f"NCC (↑): {metrics['ncc']:.4f}")
            axes[4].text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.5", fc='wheat', alpha=0.6))

            if i == 0:
                for ax, title in zip(axes, column_titles):
                    ax.set_title(title, fontsize=14, pad=10)

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

        fig.tight_layout(h_pad=3)

        save_path = os.path.join(args.output_dir, args.output_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✔️ Visualization saved to '{save_path}'")

    if all_mae:
        print("\n--- Average Evaluation Results (Full Dataset) ---")
        print(f"Average MAE (L1): {np.mean(all_mae):.4f}")
        print(f"Average SSIM:     {np.mean(all_ssim):.4f}")
        print(f"Average NCC:      {np.mean(all_ncc):.4f}")
        print("-------------------------------------------------")
    else:
        print("\nNo images were processed. Check your input directories.")

if __name__ == "__main__":
    main()
