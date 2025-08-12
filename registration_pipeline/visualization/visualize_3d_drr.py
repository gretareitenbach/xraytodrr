import pyvista
import torch
import argparse
from pathlib import Path

from diffdrr.data import read
from diffdrr.pose import RigidTransform
from diffdrr.visualization import drr_to_mesh, img_to_mesh
from xvr.renderer import initialize_drr

'''
Usage:

python visualize_3d_drr.py \
    --volume /path/to/your/volume.nii.gz \
    --parameters /path/to/your/parameters.pt \
    --output_dir /path/to/your/3d_viz_folder
    
'''

def main():
    """
    Generates a 3D visualization of the CT, detector, and camera pose for a given
    registration result.
    """
    parser = argparse.ArgumentParser(
        description="Create a 3D visualization of a DRR registration result."
    )
    parser.add_argument(
        "-v", "--volume",
        type=Path,
        required=True,
        help="Path to the CT volume file (e.g., .nii.gz)."
    )
    parser.add_argument(
        "-p", "--parameters",
        type=Path,
        required=True,
        help="Path to the .pt file containing the final registration pose."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the output render (.png) and interactive HTML (.html)."
    )
    args = parser.parse_args()  

    # Ensure the output directory exists
    args.output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Creating CT mesh from {args.volume}...")
    ct = drr_to_mesh(read(volume=args.volume), "surface_nets", threshold=225, verbose=True)
    print("CT mesh created successfully.")
    
    # Constants for DRR generation
    HEIGHT = 3362
    WIDTH = 1038
    DELX = 0.148
    DELY = 0.148
    SDD = 1020.0

    # Initialize the DRR renderer
    drr = initialize_drr(args.volume,
                mask=None,
                labels=None,
                orientation="PA",
                height=HEIGHT,
                width=WIDTH,
                sdd=SDD,
                delx=DELX,
                dely=DELY,
                x0=-0.0,
                y0=0.0,
                reverse_x_axis=True,
                renderer="siddon",
                read_kwargs={"bone_attenuation_multiplier": 3.0},
                drr_kwargs={})
    drr.rescale_detector_(0.5)
    print("DRR object initialized successfully.")

    # Load final pose
    print(f"Loading final pose from {args.parameters}...")
    final_pose = RigidTransform(torch.load(str(args.parameters), weights_only=False, map_location=device)["final_pose"])
    img = drr(final_pose)
    print("Final pose loaded successfully.")

    # Convert the DRR and pose to 3D meshes for visualization
    camera, detector, texture, principal_ray = img_to_mesh(
        drr,
        final_pose,
    )

    # Plot the 3d visualization
    print("Generating 3D visualization...")
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(ct)
    plotter.add_mesh(camera, show_edges=True)
    plotter.add_mesh(detector, texture=texture)
    plotter.add_mesh(principal_ray, color="blue")
    plotter.add_bounding_box()
    plotter.show_axes()
    plotter.show_bounds(grid='front', location='outer', all_edges=True, show_xlabels=False, show_zlabels=False)
    plotter.camera.azimuth = 180

    # Save the visualization as a static image and an interactive HTML file
    png_path = args.output_dir / "3d_render.png"
    html_path = args.output_dir / "3d_render.html"

    plotter.screenshot(str(png_path))
    plotter.export_html(str(html_path))
    plotter.close()

    print(f"✅ Visualization saved successfully!")
    print(f"  - Image: {png_path}")
    print(f"  - Interactive HTML: {html_path}")

if __name__ == "__main__":
    main()
