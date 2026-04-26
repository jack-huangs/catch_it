from typing import Union

import numpy as np
import open3d as o3d


def pcshow(point_cloud: Union[np.ndarray, list, str, o3d.geometry.PointCloud]) -> None:
    """
    Display a point cloud using Open3D.

    This function supports displaying point clouds provided as a file path to a PCD file,
    a numpy array, a list of points, or an Open3D point cloud object.

    Parameters
    ----------
    point_cloud : Union[np.ndarray, list, str, o3d.geometry.PointCloud]
        The point cloud data to display. It can be:
        - A file path to a PCD file.
        - A numpy array representing the point cloud (nx3).
        - A list of points where each point is a list or tuple of 3 coordinates.
        - An Open3D point cloud object.

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        If the provided point cloud type is unsupported.
    """
    try:
        if isinstance(point_cloud, str):
            # Load point cloud from file
            pcd = o3d.io.read_point_cloud(point_cloud)
        elif isinstance(point_cloud, (list, np.ndarray)):
            # Convert list or numpy array to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(point_cloud))
        elif isinstance(point_cloud, o3d.geometry.PointCloud):
            # Point cloud is already an Open3D point cloud object
            pcd = point_cloud
        else:
            raise ValueError("Unsupported point cloud type")

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        raise ValueError(f"Error displaying point cloud: {e}")


def pcread(filename: str) -> np.ndarray:
    """
    Read a point cloud from a PCD file and return it as a numpy array.

    Parameters
    ----------
    filename : str
        File path to the PCD file.

    Returns
    ----------
    np.ndarray
        A numpy array representing the point cloud with shape (n, 3).

    Raises
    ----------
    ValueError
        If the file cannot be read as a point cloud.
    """
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string.")

    try:
        pcd = o3d.io.read_point_cloud(filename)
        if pcd.is_empty():
            raise ValueError(f"Point cloud read from '{filename}' is empty or invalid.")
        return np.asarray(pcd.points)
    except Exception as e:
        raise ValueError(f"Error reading point cloud from '{filename}': {e}")


def pcwrite(
    filename: str, point_cloud: Union[np.ndarray, list, o3d.geometry.PointCloud]
) -> None:
    """
    Write a point cloud to a PCD file.

    Parameters
    ----------
    filename : str
        File path where the PCD file will be saved.
    point_cloud : Union[np.ndarray, list, o3d.geometry.PointCloud]
        The point cloud data to write. It can be:
        - A numpy array representing the point cloud (nx3).
        - A list of points where each point is a list or tuple of 3 coordinates.
        - An Open3D point cloud object.

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        If the provided point cloud type is unsupported or the file cannot be written.
    """
    try:
        if isinstance(point_cloud, np.ndarray):
            # Convert numpy array to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
        elif isinstance(point_cloud, list):
            # Convert list to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(point_cloud))
        elif isinstance(point_cloud, o3d.geometry.PointCloud):
            # Point cloud is already an Open3D point cloud object
            pcd = point_cloud
        else:
            raise ValueError("Unsupported point cloud type")

        # Write the point cloud to a PCD file
        o3d.io.write_point_cloud(filename, pcd)
    except Exception as e:
        raise ValueError(f"Error writing point cloud to '{filename}': {e}")


def hom(points: np.ndarray) -> np.ndarray:
    """
    Convert 3D points to homogeneous coordinates.

    Parameters
    ----------
    points : np.ndarray
        Input points organized as nx3 or 3xn.

    Returns
    ----------
    np.ndarray
        Homogenized points organized as nx4 or 4xn.

    Raises
    ----------
    ValueError
        If the input points are not organized as nx3 or 3xn.
    """
    if not isinstance(points, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if points.shape[1] == 3:
        return np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
    elif points.shape[0] == 3:
        return np.vstack((points, np.ones((1, points.shape[1]), dtype=points.dtype)))
    else:
        raise ValueError("Input points must be organized as nx3 or 3xn")


def dehom(points: np.ndarray) -> np.ndarray:
    """
    Convert homogeneous coordinates back to 3D points.

    Parameters
    ----------
    points : np.ndarray
        Input points organized as nx4 or 4xn.

    Returns
    ----------
    np.ndarray
        Dehomogenized points organized as nx3 or 3xn.

    Raises
    ----------
    ValueError
        If the input points are not organized as nx4 or 4xn.
    """

    if not isinstance(points, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    if points.shape[1] == 4:
        return points[:, :3] / points[:, 3].reshape(-1, 1)
    elif points.shape[0] == 4:
        return points[:3, :] / points[3, :].reshape(1, -1)
    else:
        raise ValueError("Input points must be organized as nx4 or 4xn")


def normalize(
    image: Union[np.ndarray, list, str, o3d.geometry.PointCloud],
) -> np.ndarray:
    """
    Normalize the input data to a range of [0, 1].

    Parameters
    ----------
    image : Union[np.ndarray, list, str, o3d.geometry.PointCloud]
        The input data to be normalized. This can be:
        - A numpy array representing an image or other data.
        - A list of data points that can be converted to a numpy array.
        - A string representing the file path to an image or point cloud file.
        - An Open3D point cloud object.

    Returns
    ----------
    np.ndarray
        The normalized data as a numpy array.

    Raises
    ----------
    ValueError
        If the input type is not supported or if an error occurs during normalization.
    """
    try:
        if isinstance(image, np.ndarray):
            # Normalize NumPy array
            return (image - np.min(image)) / (np.max(image) - np.min(image))

        elif isinstance(image, list):
            # Convert list to NumPy array and normalize
            image_array = np.array(image)
            return (image_array - np.min(image_array)) / (
                np.max(image_array) - np.min(image_array)
            )

        elif isinstance(image, str):
            # Check if the file is an image or a point cloud
            if image.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                from PIL import Image

                img = Image.open(image).convert("L")  # Convert to grayscale
                img_array = np.array(img)
                return (img_array - np.min(img_array)) / (
                    np.max(img_array) - np.min(img_array)
                )
            elif image.endswith((".ply", ".pcd", ".xyz")):
                pcd = o3d.io.read_point_cloud(image)
                pcd_array = np.asarray(pcd.points)
                return (pcd_array - np.min(pcd_array)) / (
                    np.max(pcd_array) - np.min(pcd_array)
                )
            else:
                raise ValueError("Unsupported file format")

        elif isinstance(image, o3d.geometry.PointCloud):
            # Normalize Open3D point cloud
            pcd_array = np.asarray(image.points)
            return (pcd_array - np.min(pcd_array)) / (
                np.max(pcd_array) - np.min(pcd_array)
            )

        else:
            raise ValueError("Unsupported input type")
    except Exception as e:
        raise ValueError(f"Error normalizing input data: {e}")
