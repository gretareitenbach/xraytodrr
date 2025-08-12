import imageio
import pandas as pd
import pyvista
import torch

from diffdrr.data import read
from diffdrr.pose import convert
from diffdrr.visualization import drr_to_mesh, img_to_mesh
from pathlib import Path
from tqdm import tqdm
from xvr.renderer import initialize_drr

# --- 1. Setup and Initialization ---
# Use CPU to avoid potential device mismatches in the loop
device = torch.device("cpu") 

# TODO: Update these paths if they are different
volume_path = Path("./p8/p8_tibia.nii.gz")
# This should be the path to your .pt file containing the trajectory DataFrame
trajectory_path = Path("./p8/results/xray_with_data/parameters.pt") 
save_root = Path("./results/3d_animation")
save_root.mkdir(exist_ok=True, parents=True)
animation_path = save_root / "trajectory_animation.gif"

# Create the CT mesh (only needs to be done once)
ct_mesh = drr_to_mesh(read(volume=volume_path), "surface_nets", threshold=225, verbose=True)
print("CT mesh created successfully.")

# Initialize the DRR renderer
HEIGHT = 3362
WIDTH = 1038
DELX = 0.148
DELY = 0.148

drr = initialize_drr(volume_path,
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

# --- 2. Load Trajectory Data ---
# Load the DataFrame with pose parameters
# Assuming the trajectory is stored under the key "trajectory"
trajectory_df = torch.load(trajectory_path, map_location=device, weights_only=False)["trajectory"].iloc[:, :6]
trajectory_df.columns = ['alpha', 'beta', 'gamma', 'dx', 'dy', 'dz']  # Update column names

print(f"Trajectory loaded with {len(trajectory_df)} poses.")

# convert trajectory DataFrame to tensors for efficient processing
rotations_tensor = torch.tensor(trajectory_df[['alpha', 'beta', 'gamma']].values, device=device, dtype=torch.float32)
translations_tensor = torch.tensor(trajectory_df[['dx', 'dy', 'dz']].values, device=device, dtype=torch.float32)
print("Trajectory data converted to tensors for efficient processing.")

# --- 3. Create the Animation ---
print("Starting animation generation...")
plotter = pyvista.Plotter(off_screen=True)

# Add static meshes and set up the scene view once before the loop
plotter.add_mesh(ct_mesh)

# Open the GIF file for writing
# `duration` is seconds per frame, `loop=0` means loop forever
with imageio.get_writer(animation_path, mode='I', duration=0.1, loop=0) as writer:
    # Loop through each pose in the trajectory
    for i in tqdm(range(len(trajectory_df)), desc="Generating frames"):

        # Separate rotation and translation from the DataFrame row
        rotation_params = rotations_tensor[i:i+1]
        translation_params = translations_tensor[i:i+1]

        # Convert the 6-DOF parameters to a RigidTransform object
        current_pose = convert(rotation_params, translation_params, parameterization="euler_angles", convention="XYZ")
        
        # Create the dynamic meshes for the camera and detector for this frame
        camera, detector, texture, principal_ray = img_to_mesh(drr, current_pose)
        
        # Add ONLY the dynamic meshes to the plotter, giving them unique names so they can be removed
        plotter.add_mesh(camera, name="camera", show_edges=True)
        plotter.add_mesh(detector, name="detector", texture=texture)
        plotter.add_mesh(principal_ray, name="ray", color="blue")
        plotter.add_bounding_box()
        plotter.show_axes()
        plotter.show_bounds(grid='front', location='outer', all_edges=True, show_xlabels=False, show_zlabels=False)
        plotter.camera.azimuth = 180
         
        # Grab a screenshot of the current state of the plotter
        frame = plotter.screenshot(None, return_img=True)
        writer.append_data(frame)

        # Remove the dynamic meshes to prepare for the next frame
        plotter.remove_actor("camera")
        plotter.remove_actor("detector")
        plotter.remove_actor("ray")

plotter.close()
print(f"✅ Animation saved successfully to: {animation_path}")