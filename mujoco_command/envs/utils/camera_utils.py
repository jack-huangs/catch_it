from typing import List, Union

import numpy as np
import pandas as pd
import spatialmath as sm
import spatialmath.base as smb
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def deproject_pixel_to_3d(obs, pixel, camera_name, env_cfg):
    """
    Deprojects a pixel (u, v) from a given camera into a 3D point in the robot frame.

    Args:
        obs: Observation dictionary containing image and camera parameters.
        pixel: Tuple (u, v) pixel coordinates (in image space).
        camera_name: Name of the camera in env_cfg.pcl_cameras.
        env_cfg: Environment configuration with intrinsics/extrinsics.

    Returns:
        A 3D point in the robot/world frame as a (3,) np.ndarray.
    """
    u, v = pixel  # Note: u = col (x), v = row (y)
    depth_image = obs[f'{camera_name}_depth']
    rgb_image = obs[f'{camera_name}_image']  # optional, not used here

    # Get intrinsics and extrinsics
    if env_cfg.is_sim:
        K = obs[f'{camera_name}_K']
        T = obs[f'{camera_name}_T']
        base_units = 0
    else:
        K = env_cfg.intrinsics[camera_name]
        T = env_cfg.extrinsics[camera_name]
        base_units = -3

    # Get the depth value at the pixel and scale appropriately
    depth = depth_image[v, u] * (10**base_units)  # depth is in mm in real, meters in sim

    # Back-project pixel to camera frame
    pixel_homog = np.array([u, v, 1.0])
    K_inv = np.linalg.inv(K)
    point_camera = K_inv @ pixel_homog * depth  # 3D point in camera frame

    # Convert to homogeneous coordinates and transform to robot/world frame
    point_camera_homog = np.append(point_camera, 1.0)
    point_world_homog = T @ point_camera_homog
    point_world = point_world_homog[:3]

    return point_world

def depth_to_point_cloud(depth_image, K, T, base_units=0) -> np.ndarray:
    # Get image dimensions
    dimg_shape = depth_image.shape
    height = dimg_shape[0]
    width = dimg_shape[1]

    # Create pixel grid
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Flatten arrays for vectorized computation
    x_flat = x.flatten()
    y_flat = y.flatten()
    depth_flat = depth_image.flatten()
    depth_flat = depth_flat * (10**base_units)

    # Stack flattened arrays to form homogeneous coordinates
    homogeneous_coords = np.vstack((x_flat, y_flat, np.ones_like(x_flat)))

    # Compute inverse of the intrinsic matrix K
    K_inv = np.linalg.inv(K)

    # Calculate 3D points in camera coordinates
    points_camera = np.dot(K_inv, homogeneous_coords) * depth_flat

    # Homogeneous coordinates to 3D points
    points_camera_homog = np.vstack((points_camera, np.ones_like(x_flat)))
    #points_camera_homog = camera_axis_correction @ points_camera_homog
    points_world_homog = np.dot(T, points_camera_homog)

    # dehomogenize
    points_world = points_world_homog[:3, :].T
    return points_world

def pcl_from_obs(obs, env_cfg, vis=False, cam_names=None):
    """
    Convert RGB-D observations into a merged point cloud.

    Args:
        obs: Dictionary containing RGB and depth images.
        env_cfg: RealEnvConfig object with intrinsics/extrinsics.
        vis: Whether to visualize the point cloud.
        cam_names: Optional list of camera names to use. Defaults to env_cfg.pcl_cameras.

    Returns:
        merged_points (N, 3), merged_colors (N, 3)
    """
    merged_points = []
    merged_colors = []

    cam_list = cam_names if cam_names is not None else env_cfg.pcl_cameras

    for view in cam_list:
        rgb_image = obs['%s_image' % view]
        depth_image = obs['%s_depth' % view]

        if env_cfg.is_sim:
            T = obs['%s_T' % view]
            K = obs['%s_K' % view]
            base_units = 0
        else:
            T = env_cfg.extrinsics[view]
            K = env_cfg.intrinsics[view]
            base_units = -3

        crop_min_bound = env_cfg.min_bound
        crop_max_bound = env_cfg.max_bound

        points = depth_to_point_cloud(depth_image, K, T, base_units=base_units)
        colors = rgb_image.reshape(points.shape) / 255.0

        if crop_min_bound and crop_max_bound:
            x_mask = (points[..., 0] >= crop_min_bound[0]) & (points[..., 0] <= crop_max_bound[0])
            y_mask = (points[..., 1] >= crop_min_bound[1]) & (points[..., 1] <= crop_max_bound[1])
            z_mask = (points[..., 2] >= crop_min_bound[2]) & (points[..., 2] <= crop_max_bound[2])
            xyz_mask = x_mask & y_mask & z_mask

            points = points[xyz_mask]
            colors = colors[xyz_mask]

        merged_points.append(points)
        merged_colors.append(colors)

    merged_points = np.vstack(merged_points)
    merged_colors = np.vstack(merged_colors)

    if vis:
        import open3d as o3d
        # Create an Open3D PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(merged_points)
        point_cloud.colors = o3d.utility.Vector3dVector(merged_colors)
        o3d.visualization.draw_geometries([point_cloud])

    return merged_points, merged_colors


def make_tf(
    pos: Union[np.ndarray, list] = [0, 0, 0],
    ori: Union[np.ndarray, sm.SE3, sm.SO3] = [1, 0, 0, 0],
) -> sm.SE3:
    """
    Create a SE3 transformation matrix.

    Parameters:
    - pos (np.ndarray): Translation vector [x, y, z].
    - ori (Union[np.ndarray, SE3]): Orientation, can be a rotation matrix, quaternion, or SE3 object.

    Returns:
    - SE3: Transformation matrix.
    """

    if isinstance(ori, list):
        ori = np.array(ori)

    if isinstance(ori, sm.SO3):
        ori = ori.R

    if isinstance(pos, sm.SE3):
        pose = pos
        pos = pose.t
        ori = pose.R

    if len(ori) == 9:
        ori = np.reshape(ori, (3, 3))

    # Convert ori to SE3 if it's already a rotation matrix or a quaternion
    if isinstance(ori, np.ndarray):
        if ori.shape == (3, 3):  # Assuming ori is a rotation matrix
            ori = ori
        elif ori.shape == (4,):  # Assuming ori is a quaternion
            ori = sm.UnitQuaternion(s=ori[0], v=ori[1:]).R
        elif ori.shape == (3,):  # Assuming ori is rpy
            ori = sm.SE3.Eul(ori, unit="rad").R

    T_R = smb.r2t(ori) if is_R_valid(ori) else smb.r2t(make_R_valid(ori))
    R = sm.SE3(T_R, check=False).R

    # Combine translation and orientation
    T = sm.SE3.Rt(R=R, t=pos, check=False)

    return T


def is_R_valid(R: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Checks if the input matrix R is a valid 3x3 rotation matrix.

    Parameters:
            R (np.ndarray): The input matrix to be checked.
            tol (float, optional): Tolerance for numerical comparison. Defaults to 1e-8.

    Returns:
            bool: True if R is a valid rotation matrix, False otherwise.

    Raises:
            ValueError: If R is not a 3x3 matrix.
    """
    # Check if R is a 3x3 matrix
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise ValueError(f"Input is not a 3x3 matrix. R is \n{R}")

    # Check if R is orthogonal
    is_orthogonal = np.allclose(np.dot(R.T, R), np.eye(3), atol=tol)

    # Check if the determinant is 1
    det = np.linalg.det(R)

    return is_orthogonal and np.isclose(det, 1.0, atol=tol)

def make_R_valid(R: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Makes the input matrix R a valid 3x3 rotation matrix.

    Parameters:
            R (np.ndarray): The input matrix to be made valid.
            tol (float, optional): Tolerance for numerical comparison. Defaults to 1e-6.

    Returns:
            np.ndarray: A valid 3x3 rotation matrix derived from the input matrix R.

    Raises:
            ValueError: If the input is not a 3x3 matrix or if it cannot be made a valid rotation matrix.
    """
    if not isinstance(R, np.ndarray):
        R = np.array(R)

    # Check if R is a 3x3 matrix
    if R.shape != (3, 3):
        raise ValueError("Input is not a 3x3 matrix")

    # Step 1: Gram-Schmidt Orthogonalization
    Q, _ = np.linalg.qr(R)

    # Step 2: Ensure determinant is 1
    det = np.linalg.det(Q)
    if np.isclose(det, 0.0, atol=tol):
        raise ValueError("Invalid rotation matrix (determinant is zero)")

    # Step 3: Ensure determinant is positive
    if det < 0:
        Q[:, 2] *= -1

    return Q


