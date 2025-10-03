import torch
from glob import glob
import numpy as np
import open3d as o3d

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
image_names = glob("/home/multyxu/Data/vggt/31_random_pics/*.jpg")  # Example to load multiple images from a directory
images = load_and_preprocess_images(image_names).to(device)
print("images:", images.shape, images.dtype, images.device)

with torch.no_grad():
    with torch.amp.autocast(device, dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))
    # print("point_map_by_unprojection:", point_map_by_unprojection.shape, point_map_by_unprojection.dtype, point_map_by_unprojection)

    # Predict Tracks
    # choose your own points to track, with shape (N, 2) for one scene
    query_points = torch.FloatTensor([[100.0, 200.0], 
                                        [60.72, 259.94]]).to(device)
    track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])


print("conf_score:", conf_score.shape)
print("conf_score min max:", conf_score.min(), conf_score.max())
print("images:", images.shape, images.dtype, images.device)
# Visualize the point cloud using Open3D
# 1. Create a point cloud from random data
points = point_map_by_unprojection.reshape(-1, 3)
print(f"number of points before confidence filtering: {points.shape[0]}")
conf_score = depth_conf.reshape(-1,).cpu().numpy()
print("conf_score min max:", conf_score.min(), conf_score.max())
conf_score_norm = (conf_score - conf_score.min()) / (conf_score.max() - conf_score.min())
conf_mask = conf_score_norm > 0.5
print((~conf_mask).any())
points = points[conf_mask]  # Filter points by confidence
print(f"number of points after confidence filtering: {points.shape[0]}")

color = images.squeeze().permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
color = color[conf_mask]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 2. Assign colors (optional, here we assign a uniform blue color)
pcd.colors = o3d.utility.Vector3dVector(color)

# 3. Visualize the point cloud
print("Visualizing the point cloud...")
o3d.visualization.draw_geometries([pcd])