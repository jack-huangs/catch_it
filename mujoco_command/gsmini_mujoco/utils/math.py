import math
import random
from typing import Callable, List, Tuple, Union

import numpy as np
import quaternion
import spatialmath as sm
from scipy.integrate import quad
from scipy.spatial.transform import Rotation as R
from spatialmath import UnitQuaternion

# Type alias for vectors
Vector = Union[List[float], Tuple[float, ...]]


def rotate_vector_2d(vector: Tuple[float, float], angle: float) -> np.ndarray:
    """
    Rotate a 2D vector by a given angle in radians.

    This function applies a rotation matrix to a 2D vector to achieve the rotation.

    Parameters
    ----------
    vector : Tuple[float, float]
        The 2D vector to rotate.
    angle : float
        The angle in radians to rotate the vector by.

    Returns
    -------
    np.ndarray
        The rotated 2D vector.
    """
    return np.array(
        [
            vector[0] * math.cos(angle) - vector[1] * math.sin(angle),
            vector[0] * math.sin(angle) + vector[1] * math.cos(angle),
        ]
    )


def dotproduct(v1: Vector, v2: Vector) -> float:
    """
    Calculate the dot product of two vectors.

    Parameters
    ----------
    v1 : Vector
        The first vector.
    v2 : Vector
        The second vector.

    Returns
    -------
    float
        The dot product of v1 and v2.

    Raises
    ------
    TypeError
        If either v1 or v2 is not a list or tuple of floats.
    ValueError
        If the lengths of v1 and v2 are not equal.
    """
    if not (isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple))):
        raise TypeError("Both v1 and v2 must be lists or tuples of floats.")
    if not (
        all(isinstance(i, (int, float)) for i in v1)
        and all(isinstance(i, (int, float)) for i in v2)
    ):
        raise TypeError("All elements of v1 and v2 must be floats or integers.")
    if len(v1) != len(v2):
        raise ValueError("Vectors v1 and v2 must be of the same length.")
    return sum(a * b for a, b in zip(v1, v2))


def length(v: Vector) -> float:
    """
    Calculate the length (magnitude) of a vector.

    Parameters
    ----------
    v : Vector
        The vector.

    Returns
    -------
    float
        The length of the vector.

    Raises
    ------
    TypeError
        If v is not a list or tuple of floats.
    """
    if not isinstance(v, (list, tuple)):
        raise TypeError("v must be a list or tuple of floats.")
    if not all(isinstance(i, (int, float)) for i in v):
        raise TypeError("All elements of v must be floats or integers.")
    return math.sqrt(dotproduct(v, v))


def angle(v1: Vector, v2: Vector) -> float:
    """
    Calculate the angle between two vectors in radians.

    Parameters
    ----------
    v1 : Vector
        The first vector.
    v2 : Vector
        The second vector.

    Returns
    -------
    float
        The angle between v1 and v2 in radians.

    Raises
    ------
    TypeError
        If either v1 or v2 is not a list or tuple of floats.
    ValueError
        If the lengths of v1 and v2 are not equal.
    """
    if not (isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple))):
        raise TypeError("Both v1 and v2 must be lists or tuples of floats.")
    if not (
        all(isinstance(i, (int, float)) for i in v1)
        and all(isinstance(i, (int, float)) for i in v2)
    ):
        raise TypeError("All elements of v1 and v2 must be floats or integers.")
    if len(v1) != len(v2):
        raise ValueError("Vectors v1 and v2 must be of the same length.")
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def flip(v: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Flip a vector (list or numpy array) element-wise by multiplying it with -1.

    Parameters
    ----------
    v : Union[List[float], np.ndarray]
        The vector to flip.

    Returns
    -------
    np.ndarray
        The flipped vector.
    """
    if isinstance(v, list):
        v = np.array(v)
    return v * -1


def gcd(a: float, b: float) -> float:
    """
    Compute the greatest common divisor (GCD) of two numbers using the Euclidean algorithm.

    Parameters
    ----------
    a : float
        The first number.
    b : float
        The second number.

    Returns
    -------
    float
        The GCD of a and b.
    """
    if a < b:
        return gcd(b, a)
    if abs(b) < 1e-5:
        return a
    else:
        return gcd(b, a - math.floor(a / b) * b)


def lcm(a: Union[float, List[float]], b: float = None) -> float:
    """
    Compute the least common multiple (LCM) of two numbers or a list of numbers.

    Parameters
    ----------
    a : Union[float, List[float]]
        The first number or a list of numbers.
    b : float, optional
        The second number (if a is a single number).

    Returns
    -------
    float
        The LCM of the input numbers.
    """
    if b is not None:
        return (a * b) / gcd(a, b)
    else:
        out = lcm(a[0], a[1])
        for i in range(2, len(a)):
            out = lcm(out, a[i])
        return out


def random_unit_quaternion() -> UnitQuaternion:
    """
    Generate a random unit quaternion.

    Returns
    -------
    UnitQuaternion
        A random unit quaternion.
    """
    z = 100000
    while z > 1.0:
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        z = x * x + y * y
    w = 100000
    while w > 1.0:
        u = random.uniform(-1.0, 1.0)
        v = random.uniform(-1.0, 1.0)
        w = u * u + v * v

    s = np.sqrt((1 - z) / w)
    return UnitQuaternion([x, y, s * u, s * v])


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector.

    Parameters
    ----------
    v : np.ndarray
        The vector to normalize.

    Returns
    -------
    np.ndarray
        The normalized vector.
    """
    return v / np.linalg.norm(v)


def calculate_rotation_between_vectors(
    v_from: np.ndarray, v_to: np.ndarray
) -> np.ndarray:
    """
    Calculate the rotation matrix that rotates one vector to another.

    Parameters
    ----------
    v_from : np.ndarray
        The initial vector.
    v_to : np.ndarray
        The target vector.

    Returns
    -------
    np.ndarray
        The rotation matrix.
    """
    v_from = normalize_vector(v_from)
    v_to = normalize_vector(v_to)

    angle = np.arccos(
        np.dot(v_from, v_to) / (np.linalg.norm(v_from) * np.linalg.norm(v_to))
    )
    axis = np.cross(v_from, v_to)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-5:
        if angle < 1e-5:
            return np.identity(3)
        elif np.pi - angle < 1e-5:
            axis = arbitrary_orthogonal_vector(v_from)
            axis_norm = np.linalg.norm(axis)

    axis /= axis_norm
    r = R.from_rotvec(axis * angle)
    return r.as_matrix()


def arbitrary_orthogonal_vector(vec: np.ndarray) -> np.ndarray:
    """
    Find an arbitrary vector orthogonal to the given vector.

    Parameters
    ----------
    vec : np.ndarray
        The input vector.

    Returns
    -------
    np.ndarray
        An orthogonal vector.
    """
    Ax, Ay, Az = np.abs(vec)
    if Ax < Ay:
        return (
            np.array([0, -vec[2], vec[1]])
            if Ax < Az
            else np.array([-vec[1], vec[0], 0])
        )
    else:
        return (
            np.array([vec[2], 0, -vec[0]])
            if Ay < Az
            else np.array([-vec[1], vec[0], 0])
        )


def quat_to_axang(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion or an array of quaternions to a 4D axis-angle representation.

    Parameters
    ----------
    q : np.ndarray
        A numpy array of shape (4,) representing a single quaternion (w, x, y, z)
        or of shape (N, 4) representing N quaternions.

    Returns
    -------
    np.ndarray
        A numpy array of shape (4,) representing the axis-angle of a single quaternion
        (x, y, z, angle) or of shape (N, 4) representing the axis-angle representations
        of N quaternions.
    """
    # Ensure the input is a 2D array for consistent processing
    if q.ndim == 1:
        q = np.expand_dims(q, axis=0)

    # Normalize the quaternions
    q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)

    # Compute the angle
    angles = 2 * np.arccos(np.clip(q_norm[:, 0], -1.0, 1.0))

    # Compute the axis, avoiding division by zero
    s = np.sqrt(1 - q_norm[:, 0] ** 2)
    axis = np.where(
        s[:, np.newaxis] > 1e-5, q_norm[:, 1:] / s[:, np.newaxis], q_norm[:, 1:]
    )

    # Concatenate the axis and angle to form a 4D representation
    axis_angle = np.hstack((axis, angles[:, np.newaxis]))

    # If the original input was a single quaternion, return a single axis-angle
    if axis_angle.shape[0] == 1:
        return axis_angle[0]

    return axis_angle


def conj(q: Union[np.ndarray, list, sm.UnitQuaternion]) -> np.ndarray:
    """
    Compute the conjugate of an array of quaternions.

    Parameters
    ----------
    q : Union[np.ndarray, list, sm.UnitQuaternion]
        Input quaternion(s) which can be a list, numpy array, or sm.UnitQuaternion.

    Returns
    -------
    np.ndarray
        The conjugated array of quaternions.

    Raises
    ------
    ValueError
        If the input cannot be converted to a valid quaternion array.
    """
    # Convert sm.UnitQuaternion to numpy array
    if isinstance(q, sm.UnitQuaternion):
        q = np.array([q.s, *q.v])

    # Convert list to numpy array
    if isinstance(q, list):
        q = np.array(q)

    # Ensure the array is 1D or higher and has the correct shape
    if q.ndim == 1:
        if q.shape[0] != 4:
            raise ValueError("1D quaternion must have 4 components (w, x, y, z).")
        q_conj = np.copy(q)
        q_conj[1:] *= -1
    else:
        if q.shape[-1] != 4:
            raise ValueError("Quaternions must have 4 components (w, x, y, z).")
        q_conj = np.copy(q)
        q_conj[..., 1:] *= -1

    return q_conj


def npq2np(npq: quaternion.quaternion) -> np.ndarray:
    """
    Convert a numpy quaternion to a numpy array.

    Parameters
    ----------
    npq : np.quaternion
        A quaternion from the numpy.quaternion library.

    Returns
    -------
    np.ndarray
        A numpy array with 4 elements representing the quaternion (w, x, y, z).
    """
    return np.array([npq.w, npq.x, npq.y, npq.z])


def euclidean_distance(T1: sm.SE3, T2: sm.SE3) -> float:
    """
    Calculate the Euclidean distance between the translation components of two SE3 poses.

    Parameters
    ----------
    T1 : sm.SE3
        The first SE3 pose.
    T2 : sm.SE3
        The second SE3 pose.

    Returns
    -------
    float
        The Euclidean distance between the translation components of T1 and T2.
    """
    return np.linalg.norm(T1.t - T2.t)


def angular_distance(T1: sm.SE3, T2: sm.SE3) -> float:
    """
    Calculate the angular distance between the rotation components of two SE3 poses.

    Parameters
    ----------
    T1 : sm.SE3
        The first SE3 pose.
    T2 : sm.SE3
        The second SE3 pose.

    Returns
    -------
    float
        The angular distance between the rotation components of T1 and T2 in radians.
    """
    q1 = UnitQuaternion(T1.R)
    q2 = UnitQuaternion(T2.R)
    return 2 * np.arccos(np.abs(np.dot(q1.vec, q2.vec)))


def frobenius_norm(T1: sm.SE3, T2: sm.SE3) -> float:
    """
    Calculate the Frobenius norm of the difference between two SE3 transformation matrices.

    Parameters
    ----------
    T1 : sm.SE3
        The first SE3 pose.
    T2 : sm.SE3
        The second SE3 pose.

    Returns
    -------
    float
        The Frobenius norm of the difference between the transformation matrices of T1 and T2.
    """
    return np.linalg.norm(T1.A - T2.A, "fro")


def geodesic_distance(T1: sm.SE3, T2: sm.SE3) -> float:
    """
    Calculate the geodesic distance between two SE3 poses.

    Parameters
    ----------
    T1 : sm.SE3
        The first SE3 pose.
    T2 : sm.SE3
        The second SE3 pose.

    Returns
    -------
    float
        The geodesic distance between T1 and T2.
    """
    T_diff = T1.inv() * T2
    log_map = sm.SE3.log(T_diff)
    return np.linalg.norm(log_map)


def hausdorff_distance(T1: sm.SE3, T2: sm.SE3, points: np.ndarray) -> float:
    """
    Calculate the Hausdorff distance between two SE3 poses given a set of points.

    Parameters
    ----------
    T1 : sm.SE3
        The first SE3 pose.
    T2 : sm.SE3
        The second SE3 pose.
    points : np.ndarray
        A set of points to compare the transformed poses.

    Returns
    -------
    float
        The Hausdorff distance between T1 and T2.
    """
    transformed_points_T1 = np.dot(T1.A[:3, :3], points.T).T + T1.t
    transformed_points_T2 = np.dot(T2.A[:3, :3], points.T).T + T2.t

    dist_matrix = np.linalg.norm(
        transformed_points_T1[:, np.newaxis, :]
        - transformed_points_T2[np.newaxis, :, :],
        axis=2,
    )
    hausdorff_dist = max(
        np.max(np.min(dist_matrix, axis=1)), np.max(np.min(dist_matrix, axis=0))
    )
    return hausdorff_dist


def cint(
    F: Callable[[complex], complex],
    C: Callable[[float], complex],
    dC: Callable[[float], complex],
    t_start: float = 0.0,
    t_end: float = 2 * np.pi,
    max_iter: int = 100,
):
    """
    Computes the contour integral of a complex function along a parameterized curve.

    This function performs numerical integration of a complex function F(z) along a curve C(t),
    where t is the parameter along the curve, and dC(t) is the derivative of the curve with respect
    to t. The integral is calculated from t_start to t_end using Gaussian quadrature.

    Parameters
    ----------
    F : Callable[[complex], complex]
        The complex function to integrate.
    C : Callable[[float], complex]
        A parameterized function representing the curve along which to integrate.
    dC : Callable[[float], complex]
        The derivative of the curve with respect to the parameter t.
    t_start : float, optional
        The starting value of the parameter t, by default 0.0.
    t_end : float, optional
        The ending value of the parameter t, by default 2 * np.pi.
    max_iter : int, optional
        The maximum number of iterations for the numerical integration, by default 100.

    Returns
    ----------
    tuple
        A tuple (I, I_e) where I is the result of the contour integral, and I_e is the estimated error.
    """

    def cintegral(
        t,
        F: Callable[[complex], complex],
        C: Callable[[float], complex],
        dC: Callable[[float], complex],
    ):
        """
        Computes the complex integral at a specific parameter value along a curve.

        This function evaluates the integrand at a specific parameter value `t`, which
        corresponds to a point on a curve defined by the function `C(t)`. It computes
        the product of the function `F(z)` at the point `z = C(t)` and the derivative
        of the curve `dC(t)` with respect to `t`.

        Parameters
        ----------
        t : float
            The parameter value along the curve at which to evaluate the integral.
        F : Callable[[complex], complex]
            A complex-valued function to integrate.
        C : Callable[[float], complex]
            A parameterized curve as a function of `t` that returns a complex number `z`.
        dC : Callable[[float], complex]
            The derivative of the parameterized curve `C(t)` with respect to `t`.

        Returns
        ----------
        complex
            The product of `F(z)` and the derivative `dC(t)` at the point `z = C(t)`,
            representing the integrand at `t`.
        """
        z = C(t)
        dz_dt = dC(t)
        return F(z) * dz_dt

    integral_real, real_error = quad(
        lambda t: np.real(cintegral(t, F, C, dC)),
        t_start,
        t_end,
        complex_func=True,
        limit=max_iter,
    )
    integral_imag, imag_error = quad(
        lambda t: np.imag(cintegral(t, F, C, dC)),
        t_start,
        t_end,
        complex_func=True,
        limit=max_iter,
    )
    I = integral_real + 1j * integral_imag
    I_e = real_error + 1j * imag_error
    return I, I_e


def homotopy_class(F: Callable[[complex], complex], points: np.ndarray):
    """
    Computes the homotopy class of a curve by integrating a complex function along the curve.

    This function numerically evaluates the contour integral of a complex function F along a curve
    defined by a set of 2D points. The integral result provides information about the homotopy class
    of the curve with respect to the function F.

    Parameters
    ----------
    F : Callable[[complex], complex]
        The complex function to integrate along the curve.
    points : np.ndarray
        A NumPy array of shape (N, 2) representing the curve as a sequence of 2D points.

    Returns
    ----------
    tuple
        A tuple (I, I_e) where I is the result of the contour integral, and I_e is the estimated error.
    """

    t_start = 0
    t_end = 1

    def C(t: float) -> complex:
        """
        Returns a complex number representing a point on a piecewise linear curve.

        This function computes the position of a point on a piecewise linear curve
        parameterized by `t`. The curve is defined by a sequence of points in a
        2D plane, and `t` is used to interpolate between these points. If `t` is
        less than or equal to 0, the function returns the first point on the curve.
        If `t` is greater than or equal to 1, the function returns the last point.

        Parameters
        ----------
        t : float
            A parameter value between 0 and 1 representing a point along the curve.
            Values less than 0 return the first point, and values greater than 1
            return the last point.

        Returns
        ----------
        complex
            A complex number representing the interpolated point on the curve, where
            the real part corresponds to the x-coordinate and the imaginary part
            corresponds to the y-coordinate.
        """
        if t <= 0:
            return complex(real=points[0][0], imag=points[0][1])
        elif t >= 1:
            return complex(real=points[-1][0], imag=points[-1][1])

        segment_index = int(t * (len(points) - 1))
        p0 = points[segment_index]
        p1 = points[segment_index + 1]
        t_local = (t * (len(points) - 1)) - segment_index

        x = p0[0] + (p1[0] - p0[0]) * t_local
        y = p0[1] + (p1[1] - p0[1]) * t_local

        return complex(real=x, imag=y)

    def dC(t: float, delta_t: float = 1e-6) -> complex:
        """
        Computes the derivative of a parameterized curve at a given point.

        This function calculates the numerical derivative of the parameterized
        curve `C(t)` at a specific point `t` using central difference approximation.
        The derivative represents the rate of change of the curve with respect to
        `t`, returning a complex number where the real part represents the x-component
        of the derivative and the imaginary part represents the y-component.

        Parameters
        ----------
        t : float
            The parameter value at which to compute the derivative along the curve.
        delta_t : float, optional
            The step size used for central difference approximation (default is 1e-6).

        Returns
        ----------
        complex
            A complex number representing the derivative of the curve at `t`, where
            the real part corresponds to the derivative of the x-coordinate, and the
            imaginary part corresponds to the derivative of the y-coordinate.
        """
        next_point = C(t + delta_t)
        prev_point = C(t - delta_t)
        dz_dt = (next_point - prev_point) / (2 * delta_t)
        return dz_dt

    I, I_e = cint(F, C, dC, t_start, t_end)
    return I, I_e
