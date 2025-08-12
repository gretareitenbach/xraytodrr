# Visualization

These scripts help visualize the registration results in both 2D and 3D, which is useful for debugging, analysis, and creating figures.

-----

### Create 3D Visualization

**Purpose**: This script generates a 3D scene that shows the relationship between the CT volume, the X-ray detector, and the camera's final pose. It produces both a static image and an interactive HTML file.

**Script**: `visualize_3d_drr.py`

**Usage**:
Provide the CT volume, the final pose parameters file, and an output directory.

```bash
python visualize_3d_drr.py \
    --volume /path/to/your/volume.nii.gz \
    --parameters /path/to/your/parameters.pt \
    --output_dir /path/to/your/3d_viz_folder
```

<img width="1024" height="768" alt="3d_render" src="https://github.com/user-attachments/assets/9054b8c0-2ad4-4fdc-88aa-eb19b126c29d" />

-----

### Animate 2D Registration

**Purpose**: This script creates an animated GIF that shows the iterative 2D registration process, visualizing how the DRR aligns with the target X-ray over time.

**Script**: `animate_2d_pose.py`

**Usage**:
Provide the CT volume, the parameters file containing the registration trajectory, and an output path for the GIF.

![2d_animation](https://github.com/user-attachments/assets/4e008061-4be0-4bf9-8546-9945a35a8f22)

```bash
python animate_2d_pose.py \
    --volume /path/to/your/ct_scan.nii.gz \
    --parameters /path/to/registration_output/final_pose.pt \
    --output_path /path/to/save/registration_animation.gif
```

-----

### Animate 3D Registration

**Purpose**: This script creates an animated GIF that shows the iterative 3D registration process, visualizing how the camera pose aligns with the target X-ray over time.

**Script**: `animate_3d_pose.py`

**Usage**:
Provide the CT volume, the parameters file containing the registration trajectory, and an output path for the GIF and HTML files.

![trajectory_animation](https://github.com/user-attachments/assets/87c779a1-e432-48eb-bac6-a0b704b7cee5)

```bash
python animate_3d_pose.py \
    --volume /path/to/your/ct_scan.nii.gz \
    --parameters /path/to/registration_output/paraemters.pt \
    --output_path /path/to/save/registration_animation.gif
```
