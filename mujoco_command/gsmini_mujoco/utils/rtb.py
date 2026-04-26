from typing import List, Union

import numpy as np
import pandas as pd
import spatialmath as sm
import spatialmath.base as smb
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import csv


def make_tf(
    pos: Union[np.ndarray, list] = [0, 0, 0],
    ori: Union[np.ndarray, sm.SE3, sm.SO3] = [1, 0, 0, 0],
) -> sm.SE3:
    """
    Create an SE3 transformation matrix from the provided position and orientation.

    This function constructs a SE3 transformation matrix that combines a translation vector
    and an orientation. The orientation can be specified in various formats including rotation
    matrices, quaternions, or SE3 objects. The function handles conversion between these formats
    and constructs the final SE3 transformation matrix.

    Parameters
    ----------
    pos : Union[np.ndarray, list], optional
        The translation vector as a list or ndarray with shape (3,). Defaults to [0, 0, 0].
    ori : Union[np.ndarray, sm.SE3, sm.SO3], optional
        The orientation can be a rotation matrix (3x3), quaternion (4,), or SE3 object.
        Defaults to [1, 0, 0, 0].

    Returns
    ----------
    sm.SE3
        The resulting SE3 transformation matrix combining the provided position and orientation.

    Notes
    -----
    - The function handles various input formats for orientation and performs necessary conversions.
    - The position and orientation must be compatible with SE3 transformation.
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
    Check if the given matrix is a valid 3x3 rotation matrix.

    This function verifies that the provided matrix is a valid rotation matrix by checking
    its orthogonality and ensuring that its determinant is close to 1. The function uses a
    tolerance level to account for numerical inaccuracies.

    Parameters
    ----------
    R : np.ndarray
        The matrix to be checked.
    tol : float, optional
        Tolerance for numerical comparison. Defaults to 1e-8.

    Returns
    ----------
    bool
        True if the matrix is a valid rotation matrix, False otherwise.

    Raises
    ------
    ValueError
        If the input matrix is not 3x3.

    Notes
    -----
    - The function performs orthogonality check and determinant check.
    """
    # Check if R is a 3x3 matrix
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise ValueError(f"Input is not a 3x3 matrix. R is \n{R}")

    # Check if R is orthogonal
    is_orthogonal = np.allclose(np.dot(R.T, R), np.eye(3), atol=tol)

    # Check if the determinant is 1
    det = np.linalg.det(R)

    return is_orthogonal and np.isclose(det, 1.0, atol=tol)


def is_ori_valid(ori: Union[np.ndarray, sm.SE3] = [1, 0, 0, 0]) -> bool:
    """
    Check if the input orientation representation is valid.

    This function verifies if the provided orientation is valid, which can be in the form of
    a rotation matrix, quaternion, or Euler angles. It checks the validity of the rotation
    matrix derived from these representations.

    Parameters
    ----------
    ori : Union[np.ndarray, sm.SE3], optional
        The orientation representation to be checked. Could be a rotation matrix (3x3),
        quaternion (4,), or Euler angles (3,). Defaults to [1, 0, 0, 0].

    Returns
    ----------
    bool
        True if the orientation representation is valid, False otherwise.

    Raises
    ------
    ValueError
        If the input orientation is not of a recognized format or invalid dimensions.

    Notes
    -----
    - The function performs conversion to rotation matrix if necessary and validates it.
    """
    if isinstance(ori, np.ndarray):
        if ori.shape == (3, 3):  # Assuming ori is a rotation matrix
            R = ori
        elif ori.shape == (4,):  # Assuming ori is a quaternion
            R = sm.UnitQuaternion(s=ori[0], v=ori[1:]).R
        elif ori.shape == (3,):  # Assuming ori is Euler angles
            R = sm.SE3.Eul(ori, unit="rad").R
        else:
            raise ValueError(f"Invalid array shape for orientation: {ori.shape}")
    elif isinstance(ori, sm.SE3):
        R = ori.R
    else:
        raise ValueError("Unsupported type for orientation")

    return is_R_valid(R)


def make_R_valid(R: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Make the input matrix a valid 3x3 rotation matrix.

    This function corrects the input matrix to ensure it is a valid rotation matrix. It
    uses Gram-Schmidt orthogonalization and adjusts the determinant to be positive.

    Parameters
    ----------
    R : np.ndarray
        The matrix to be corrected.
    tol : float, optional
        Tolerance for numerical comparison. Defaults to 1e-6.

    Returns
    ----------
    np.ndarray
        A valid 3x3 rotation matrix derived from the input matrix.

    Raises
    ------
    ValueError
        If the input matrix cannot be made a valid rotation matrix.

    Notes
    -----
    - The function performs orthogonalization and adjusts the determinant if necessary.
    """

    if is_R_valid(R):
        return R

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


def csvread(
    path: str,
    headers: List[str] = [
        "target_TCP_pose_0",
        "target_TCP_pose_1",
        "target_TCP_pose_2",
        "target_TCP_pose_3",
        "target_TCP_pose_4",
        "target_TCP_pose_5",
    ],
) -> sm.SE3:
    """
    Read a CSV file and build SE3 homogeneous transformation matrices.

    This function reads a CSV file containing columns for translation and rotation (in roll,
    pitch, yaw) and constructs a sequence of SE3 transformation matrices from the data.

    Parameters
    ----------
    path : str
        The path to the CSV file containing the transformation data.
    headers : List[str], optional
        The headers of the CSV file columns that correspond to the translation and rotation data.
        Defaults to a predefined set of headers.

    Returns
    ----------
    sm.SE3
        A sequence of SE3 transformation matrices built from the CSV data.

    Raises
    ------
    ValueError
        If the CSV file does not contain the required columns.

    Notes
    -----
    - The CSV file must include columns specified in the `headers` parameter.
    """
    # Read the CSV file
    df = pd.read_csv(path)

    # Ensure the required columns are present
    if not all(col in df.columns for col in headers):
        raise ValueError(f"CSV file must contain the following columns: {headers}")

    # Build SE3 transformation matrices
    transformations = []
    for _, row in df.iterrows():
        # Create a transformation matrix from translation and rotation
        translation = [row[headers[0]], row[headers[1]], row[headers[2]]]
        rotation = smb.trnorm(
            smb.rpy2r([row[headers[3]], row[headers[4]], row[headers[5]]], order="xyz")
        )
        transformations.append(sm.SE3.Rt(rotation, translation))

    return sm.SE3(transformations)


def trapezoidal_times(steps, accel_ratio=0.1, decel_ratio=0.1):
    """
    Generate trapezoidal profile timing steps.

    This function creates a time array that represents a trapezoidal velocity profile, including
    acceleration, constant velocity, and deceleration phases.

    Parameters
    ----------
    steps : int
        The total number of steps for the trapezoidal profile.
    accel_ratio : float, optional
        The ratio of the acceleration phase duration to the total duration. Defaults to 0.1.
    decel_ratio : float, optional
        The ratio of the deceleration phase duration to the total duration. Defaults to 0.1.

    Returns
    ----------
    np.ndarray
        An array of times normalized to [0, 1] representing the trapezoidal profile.

    Notes
    -----
    - The time array includes segments for acceleration, constant velocity, and deceleration.
    """
    accel_steps = int(accel_ratio * steps)
    decel_steps = int(decel_ratio * steps)
    const_steps = steps - accel_steps - decel_steps

    # Generate time steps for each phase
    accel_times = np.linspace(0, accel_ratio, accel_steps, endpoint=False)
    const_times = np.linspace(accel_ratio, 1 - decel_ratio, const_steps, endpoint=False)
    decel_times = np.linspace(1 - decel_ratio, 1, decel_steps)

    # Concatenate the times
    times = np.concatenate((accel_times, const_times, decel_times))
    return times


def ctraj(
    T_start: sm.SE3,
    T_end: sm.SE3,
    t: Union[int, List[float]],
    accel_ratio: float = 0.1,
    decel_ratio: float = 0.1,
) -> List[sm.SE3]:
    """
    Interpolate between two SE3 homogeneous transformation matrices.

    This function generates a list of SE3 transformation matrices by interpolating between
    a start and end transformation matrix over a specified set of time steps or a defined number of steps.
    If a list of time steps is provided, the function uses those directly; otherwise, it generates
    a trapezoidal profile based on the number of steps.

    Parameters
    ----------
    T_start : sm.SE3
        The starting SE3 transformation matrix.
    T_end : sm.SE3
        The ending SE3 transformation matrix.
    t : Union[int, List[float]]
        Either an integer specifying the number of steps, or a list of time steps for interpolation.
    accel_ratio : float, optional
        The ratio of the acceleration phase duration to the total duration when using trapezoidal profile. Defaults to 0.1.
    decel_ratio : float, optional
        The ratio of the deceleration phase duration to the total duration when using trapezoidal profile. Defaults to 0.1.

    Returns
    ----------
    List[sm.SE3]
        A list of interpolated SE3 transformation matrices.

    Notes
    -----
    - Interpolation includes both translation and rotation components.
    - The function uses spherical linear interpolation (SLERP) for rotation.
    - If `t` is an integer, a trapezoidal profile is used to generate the time steps.
    """
    # Extract translation components
    t_start = T_start.t
    t_end = T_end.t

    # Extract rotation components and convert to numpy arrays
    R_start = T_start.R
    R_end = T_end.R

    # Determine time steps
    if isinstance(t, int):
        # Generate trapezoidal time steps if `t` is an integer
        t_array = trapezoidal_times(t, accel_ratio, decel_ratio)
    else:
        # Use the provided list of time steps
        t_array = np.array(t)

    # Initialize list of interpolated transformations
    interpolated_trajectories = []

    # Normalize time steps for interpolation
    normalized_t = (t_array - t_array[0]) / (t_array[-1] - t_array[0])

    # Create SLERP object
    slerp = Slerp([0, 1], R.from_matrix([R_start, R_end]))

    for alpha in normalized_t:
        # Interpolate translations
        t_interp = (1 - alpha) * t_start + alpha * t_end

        # Interpolate rotations using SLERP
        R_interp = slerp([alpha]).as_matrix()[0]

        # Create the interpolated SE3 transformation matrix
        T_interp = sm.SE3.Rt(R_interp, t_interp)

        # Add to the list
        interpolated_trajectories.append(T_interp)

    return interpolated_trajectories


def save_traj(Traj: List[sm.SE3], save_path: str) -> None:
    """
    Save a list of SE3 transformation matrices to a CSV file with labeled columns.

    This function converts a list of SE3 objects into 4x4 numpy arrays, flattens them,
    and saves them to a CSV file with each rotational component labeled as r11, r12, ..., r33,
    and each translational component labeled as t1, t2, t3.

    Parameters
    ----------
    Traj : List[sm.SE3]
        A list of SE3 objects representing the trajectory to be saved.
    save_path : str
        The file path where the trajectory will be saved as a CSV file.

    Returns
    ----------
    None

    Raises
    ------
    TypeError
        If Traj is not a list of SE3 objects or save_path is not a string.
    """
    if not isinstance(Traj, list) or not all(isinstance(se3, sm.SE3) for se3 in Traj):
        raise TypeError("Traj must be a list of SE3 objects")
    if not isinstance(save_path, str):
        raise TypeError("save_path must be a string")

    # Define the CSV headers for the 4x4 transformation matrix (excluding the last row of [0, 0, 0, 1])
    headers = [f"r{i}{j}" for i in range(1, 4) for j in range(1, 4)] + [
        "t1",
        "t2",
        "t3",
    ]

    # Convert the SE3 objects to 4x4 numpy arrays and extract the relevant parts
    flattened_matrices = []
    for se3 in Traj:
        rotation = se3.R.flatten()  # Extract rotation part (3x3)
        translation = se3.t.flatten()  # Extract translation part (3x1)
        flattened_matrices.append(np.concatenate((rotation, translation)))

    # Save to a CSV file
    with open(save_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the headers
        writer.writerows(flattened_matrices)  # Write the flattened matrices


def load_traj(load_path: str) -> List[sm.SE3]:
    """
    Load a trajectory from a CSV file with labeled columns and return it as a list of SE3 objects.

    This function reads a CSV file containing a sequence of flattened SE3 transformation matrices
    (with labeled columns) and converts them into a list of SE3 objects representing the trajectory.

    Parameters
    ----------
    load_path : str
        The file path from which the trajectory will be loaded.

    Returns
    ----------
    List[sm.SE3]
        A list of SE3 objects representing the loaded trajectory.

    Raises
    ------
    TypeError
        If load_path is not a string.
    ValueError
        If the file content cannot be correctly interpreted as SE3 matrices.

    Notes
    -----
    - Ensure that load_path points to a valid CSV file containing the expected format of flattened SE3 matrices.
    """
    if not isinstance(load_path, str):
        raise TypeError("load_path must be a string")

    # Load the data from the CSV file
    try:
        with open(load_path, mode="r") as file:
            reader = csv.reader(file)
            headers = next(reader)  # Skip headers
            data = np.array([row for row in reader], dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to load data from {load_path}: {str(e)}")

    # Check if the number of columns is 12 (9 for rotation + 3 for translation)
    if data.shape[1] != 12:
        raise ValueError(
            "The file does not contain valid SE3 matrices with correct rotation and translation components."
        )

    # Convert the loaded data into a list of SE3 objects
    Traj = []
    for row in data:
        rotation = row[:9].reshape(3, 3)  # First 9 elements for rotation
        translation = row[9:].reshape(3, 1)  # Last 3 elements for translation
        matrix = np.eye(4)  # Create a 4x4 identity matrix
        matrix[:3, :3] = rotation  # Insert rotation part
        matrix[:3, 3] = translation.flatten()  # Insert translation part
        Traj.append(sm.SE3(matrix))

    return Traj


def cubic_interpolation(
    t: float, t0: float, tf: float, q0: np.ndarray, qf: np.ndarray
) -> np.ndarray:
    """
    Compute cubic interpolation between two joint configurations.

    This function generates an interpolated joint configuration at a given time `t`
    using cubic interpolation. It ensures a smooth transition between the initial and
    final configurations over the specified time interval.

    Parameters
    ----------
    t : float
        The current time at which to compute the joint configuration.
    t0 : float
        The start time of the trajectory.
    tf : float
        The end time of the trajectory.
    q0 : np.ndarray
        The initial joint configuration as a 1D array.
    qf : np.ndarray
        The final joint configuration as a 1D array.

    Returns
    ----------
    np.ndarray
        The interpolated joint configuration at time `t`.

    Notes
    -----
    - The function uses Hermite cubic polynomials for smooth interpolation.
    - The input joint configurations `q0` and `qf` must have the same dimensions.
    """
    tau = (t - t0) / (tf - t0)
    tau2 = tau * tau
    tau3 = tau2 * tau

    h00 = 2 * tau3 - 3 * tau2 + 1
    h10 = tau3 - 2 * tau2 + tau
    h01 = -2 * tau3 + 3 * tau2
    h11 = tau3 - tau2

    q = h00 * q0 + h01 * qf
    return q


def jtraj(q0: np.ndarray, qf: np.ndarray, t_array: np.ndarray) -> np.ndarray:
    """
    Generate a joint space trajectory using cubic interpolation.

    This function creates a smooth trajectory between two joint configurations over a specified
    time array. It generates a sequence of joint configurations by applying cubic interpolation
    for each time step in the array.

    Parameters
    ----------
    q0 : np.ndarray
        The initial joint configuration as a 1D array.
    qf : np.ndarray
        The final joint configuration as a 1D array.
    t_array : np.ndarray
        An array of time steps over which the trajectory is computed.

    Returns
    ----------
    np.ndarray
        A 2D array where each row represents a joint configuration at a corresponding time step.

    Notes
    -----
    - The initial and final joint configurations `q0` and `qf` must have the same dimensions.
    - The function assumes uniform time steps within `t_array`.
    """
    traj = np.zeros((len(t_array), len(q0)))

    for i, t in enumerate(t_array):
        traj[i, :] = cubic_interpolation(t, t_array[0], t_array[-1], q0, qf)

    return traj
