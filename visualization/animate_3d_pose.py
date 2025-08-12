import imageio
import pandas as pd
import pyvista
import torch
import argparse

from diffdrr.data import read
from diffdrr.pose import convert
from diffdrr.visualization import drr_to_mesh, img_to_mesh
from pathlib import Path
from tqdm import tqdm
from xvr.renderer import initialize_drr


def main():

    parser = argparse.ArgumentParser(
        description="Generate a DRR from a CT volume and a pose file."
    )
    parser.add_argument(
        "--volume",
        type=Path,
        required=True,
        help="Path to the input CT volume file (e.g., .nii.gz)."
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        required=True,
        help="Path to the input parameters file outputted by xvr (e.g., .pt)."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the output animation (e.g., ./results/3d_animation)."
    )
    
    args = parser.parse_args()

    device = torch.device("cpu") 

    save_root = args.output_path
    save_root.mkdir(exist_ok=True, parents=True)
    animation_path = save_root / "trajectory_animation.gif"

    # Create CT mesh
    ct_mesh = drr_to_mesh(read(volume=args.volume), "surface_nets", threshold=225, verbose=True)
    print("CT mesh created successfully.")

    # Initialize DRR renderer
    HEIGHT = 3362
    WIDTH = 1038
    DELX = 0.148
    DELY = 0.148

    drr = initialize_drr(args.volume,
                mask=None,
                labels=None,
                orientation="PA",
                height=HEIGHT,
                width=WIDTH,
                sdd=1020.0,
                delx=DELX,
                dely=DELY,
                x0=-0.0,
                y0=0.0,
                reverse_x_axis=True,
                renderer="siddon",
                read_kwargs={"bone_attenuation_multiplier": 3.0},
                drr_kwargs={}).to(device)
    drr.rescale_detector_(0.5)
    print("DRR object initialized successfully.")

    # Load the trajectory parameters
    trajectory_df = torch.load(args.parameters, map_location=device, weights_only=False)["trajectory"].iloc[:, :6]
    trajectory_df.columns = ['alpha', 'beta', 'gamma', 'dx', 'dy', 'dz']  # Update column names

    print(f"Trajectory loaded with {len(trajectory_df)} poses.")

    # convert trajectory DataFrame to tensors
    rotations_tensor = torch.tensor(trajectory_df[['alpha', 'beta', 'gamma']].values, device=device, dtype=torch.float32)
    translations_tensor = torch.tensor(trajectory_df[['dx', 'dy', 'dz']].values, device=device, dtype=torch.float32)
    print("Trajectory data converted to tensors.")

    # Create the animation 
    print("Starting animation generation...")
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(ct_mesh)

    with imageio.get_writer(animation_path, mode='I', duration=0.1, loop=0) as writer:
        # Loop through each pose in the trajectory
        for i in tqdm(range(len(trajectory_df)), desc="Generating frames"):

            rotation_params = rotations_tensor[i:i+1]
            translation_params = translations_tensor[i:i+1]
            current_pose = convert(rotation_params, translation_params, parameterization="euler_angles", convention="XYZ")

            camera, detector, texture, principal_ray = img_to_mesh(drr, current_pose)
            plotter.add_mesh(camera, name="camera", show_edges=True)
            plotter.add_mesh(detector, name="detector", texture=texture)
            plotter.add_mesh(principal_ray, name="ray", color="blue")
            plotter.add_bounding_box()
            plotter.show_axes()
            plotter.show_bounds(grid='front', location='outer', all_edges=True, show_xlabels=False, show_zlabels=False)
            plotter.camera.azimuth = 180

            frame = plotter.screenshot(None, return_img=True)
            writer.append_data(frame)

            plotter.remove_actor("camera")
            plotter.remove_actor("detector")
            plotter.remove_actor("ray")

    plotter.close()
    print(f"✅ Animation saved successfully to: {animation_path}")

if __name__ == "__main__":
    main()