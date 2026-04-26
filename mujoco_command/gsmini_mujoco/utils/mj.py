import os
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import List, Optional, Tuple, Union
from xml.dom import minidom

import mujoco as mj
import numpy as np
import spatialmath as sm
import spatialmath.base as smb
from mujoco.viewer import Handle

from utils.rtb import make_tf


@dataclass
class ContactState:
    """
    A class to represent the state of a contact in a MuJoCo simulation.

    Attributes
    ----------
    - H : np.ndarray
        A 36-element cone Hessian, set by `mj_updateConstraint`.
    - dim : int
        The contact space dimensionality: 1, 3, 4, or 6.
    - dist : float
        The distance between the nearest points; negative values indicate penetration.
    - efc_address : int
        The address in the constraint force Jacobian.
    - elem : np.ndarray
        A 2-element array of integers representing element IDs; -1 for geom or flex vertex.
    - exclude : int
        Exclusion flag for the contact: 0 (include), 1 (in gap), 2 (fused), 3 (no dofs).
    - flex : np.ndarray
        A 2-element array of integers representing flex IDs; -1 for geom.
    - frame : np.ndarray
        A 9-element array representing the contact frame. The normal is in [0-2] and points from geom[0] to geom[1].
    - friction : np.ndarray
        A 5-element array representing friction parameters: tangent1, 2, spin, roll1, roll2.
    - geom : np.ndarray
        A 2-element array of integers representing the IDs of the geometries in contact; -1 for flex.
    - geom1 : int
        The first geometry index.
    - geom2 : int
        The second geometry index.
    - includemargin : float
        The inclusion margin for the contact; includes if dist < includemargin = margin - gap.
    - mu : float
        The coefficient of friction.
    - pos : np.ndarray
        A 3-element array representing the position of the contact point, typically the midpoint between geometries.
    - solimp : np.ndarray
        A 5-element array for solver impedance parameters.
    - solref : np.ndarray
        A 2-element array for solver reference parameters for the normal direction.
    - solreffriction : np.ndarray
        A 2-element array for solver reference parameters related to friction directions.
    - vert : np.ndarray
        A 2-element array of integers representing vertex IDs; -1 for geom or flex element.
    - index : int
        The index of the contact in the simulation.
    - model : mj.MjModel
        The MuJoCo model object.
    - data : mj.MjData
        The MuJoCo data object.
    - wrench : np.ndarray
        A 6-element array representing the wrench (force and torque) at the contact.

    Methods
    -------
    - _compute_wrench():
        Computes the wrench for the contact based on the model, data, and contact index.

    - from_mjcontact(cls, mjcontact, model, data, index=-1):
        Creates a ContactState instance from a given MjContact object.
    """

    H: np.ndarray = field(default_factory=lambda: np.zeros(36))
    dim: int = 0
    dist: float = 0.0
    efc_address: int = 0
    elem: np.ndarray = field(default_factory=lambda: np.array([-1, -1], dtype=int))
    exclude: int = 0
    flex: np.ndarray = field(default_factory=lambda: np.array([-1, -1], dtype=int))
    frame: np.ndarray = field(default_factory=lambda: np.zeros(9))
    friction: np.ndarray = field(default_factory=lambda: np.zeros(5))
    geom: np.ndarray = field(default_factory=lambda: np.array([-1, -1], dtype=int))
    geom1: int = 0
    geom2: int = 0
    includemargin: float = 0.0
    mu: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    solimp: np.ndarray = field(default_factory=lambda: np.zeros(5))
    solref: np.ndarray = field(default_factory=lambda: np.zeros(2))
    solreffriction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vert: np.ndarray = field(default_factory=lambda: np.array([-1, -1], dtype=int))
    index: int = -1
    data: mj.MjData = None
    model: mj.MjModel = None
    wrench: np.ndarray = field(init=False)

    def __post_init__(self):
        self.wrench = self._compute_wrench()

    def _compute_wrench(self):
        c_array = np.zeros(6, dtype=np.float64)
        if self.index >= 0 and self.model is not None and self.data is not None:
            mj.mj_contactForce(self.model, self.data, self.index, c_array)
        return c_array

    @classmethod
    def from_mjcontact(
        cls, mjcontact, data: mj.MjData, model: mj.MjModel, index: int = -1
    ):
        return cls(
            H=mjcontact.H,
            dim=mjcontact.dim,
            dist=mjcontact.dist,
            efc_address=mjcontact.efc_address,
            elem=mjcontact.elem,
            exclude=mjcontact.exclude,
            flex=mjcontact.flex,
            frame=mjcontact.frame,
            friction=mjcontact.friction,
            geom=mjcontact.geom,
            geom1=mjcontact.geom1,
            geom2=mjcontact.geom2,
            includemargin=mjcontact.includemargin,
            mu=mjcontact.mu,
            pos=mjcontact.pos,
            solimp=mjcontact.solimp,
            solref=mjcontact.solref,
            solreffriction=mjcontact.solreffriction,
            vert=mjcontact.vert,
            index=index,
            data=data,
            model=model,
        )


class RobotInfo:
    def __init__(self, data: mj.MjData, model: mj.MjModel, name: str):
        """
        Initialize the RobotInfo class with information from the MuJoCo model.

        Parameters
        ----------
        data : mj.MjData
            The MuJoCo data object containing the current state of the simulation.
        model : mj.MjModel
            The MuJoCo model object containing the simulation model.
        name : str
            The name of the robot to retrieve information for.

        Raises
        ------
        ValueError
            If the base body for the model does not exist.
        """
        self._data = data
        self._model = model
        self.name = name

        # Retrieve and store model information
        self._body_ids = get_model_body_ids(model, name)
        self._body_names = get_model_body_names(model, name)

        does_base_exist, self._base_body_name = get_base_body_name(data, model, name)
        if not does_base_exist:
            raise ValueError(
                f"The base body for the model '{name}' does not exist. "
                f"Ensure that the base body name contains both 'base' and the model's name '{name}'."
            )

        self._actuator_ids = get_model_actuator_ids(model, name)
        self._actuator_names = [
            actuator_id2name(model, aid) for aid in self._actuator_ids
        ]

        self._geom_ids = get_model_geom_ids(model, name)
        self._geom_names = get_model_geom_names(model, name)

        self._joint_indxs = get_model_joint_qpos_indxs(data, model, name)
        self._dof_indxs = get_model_joint_dof_indxs(model, name)
        self._joint_ids = get_model_joint_ids(model, name)

        self._joint_names = get_model_joint_names(model, name)

    def print(self) -> None:
        """
        Print the robot's information in a formatted and indented list.

        This method outputs the robot's detailed information, including base body name,
        number of bodies, actuators, joints, geometries, joint limits, and actuator limits.
        Each component's name and ID are listed with proper indentation to provide a clear,
        organized view of the robot's structure and components.
        """
        print(f"Robot Name: {self.name}")
        print(f"Base Body Name: {self.base_body_name}")
        print(f"Number of Bodies: {len(self.body_names)}")
        print(f"Number of Actuators: {self.n_actuators}")
        print(f"Number of Joints: {self.n_joints}")
        print(f"Number of Geometries: {len(self.geom_names)}\n")

        print("Bodies:")
        for i, body_name in enumerate(self.body_names):
            print(f"  {i + 1}. {body_name} (ID: {self.body_ids[i]})")

        print("\nActuators:")
        for i, actuator_name in enumerate(self.actuator_names):
            print(f"  {i + 1}. {actuator_name} (ID: {self.actuator_ids[i]})")

        print("\nJoints:")
        for i, joint_name in enumerate(self.joint_names):
            print(f"  {i + 1}. {joint_name} (ID: {self.joint_ids[i]})")

        print("\nJoint Limits (min, max):")
        for i, (joint_name, limits) in enumerate(
            zip(self.joint_names, self.joint_limits.T)
        ):
            print(f"  {i + 1}. {joint_name}: {limits[0]:.2f}, {limits[1]:.2f}")

        print("\nActuator Limits (min, max):")
        for i, (actuator_name, limits) in enumerate(
            zip(self.actuator_names, self.actuator_limits.T)
        ):
            print(f"  {i + 1}. {actuator_name}: {limits[0]:.2f}, {limits[1]:.2f}")

        print("\nGeometries:")
        for i, geom_name in enumerate(self.geom_names):
            print(f"  {i + 1}. {geom_name} (ID: {self.geom_ids[i]})")

    @property
    def body_ids(self):
        """
        List of body IDs associated with the robot model.

        Returns
        -------
        List[int]
            A list of integer IDs representing the bodies in the robot model.
        """
        return self._body_ids

    @property
    def body_names(self):
        """
        List of body names associated with the robot model.

        Returns
        -------
        List[str]
            A list of names of the bodies in the robot model.
        """
        return self._body_names

    @property
    def base_body_name(self):
        """
        Name of the base body of the robot model.

        Returns
        -------
        str
            The name of the base body if it exists, otherwise an empty string.
        """
        return self._base_body_name

    @property
    def actuator_ids(self):
        """
        List of actuator IDs associated with the robot model.

        Returns
        -------
        List[int]
            A list of integer IDs representing the actuators in the robot model.
        """
        return self._actuator_ids

    @property
    def actuator_names(self):
        """
        List of actuator names associated with the robot model.

        Returns
        -------
        List[str]
            A list of names of the actuators in the robot model.
        """
        return self._actuator_names

    @property
    def geom_ids(self):
        """
        List of geometry IDs associated with the robot model.

        Returns
        -------
        List[int]
            A list of integer IDs representing the geometries in the robot model.
        """
        return self._geom_ids

    @property
    def geom_names(self):
        """
        List of geometry names associated with the robot model.

        Returns
        -------
        List[str]
            A list of names of the geometries in the robot model.
        """
        return self._geom_names

    @property
    def joint_indxs(self):
        """
        List of indices for joint positions in the robot model.

        Returns
        -------
        List[int]
            A list of integer indices corresponding to the joint positions in the robot model.
        """
        return self._joint_indxs

    @property
    def dof_indxs(self):
        """
        List of indices for degrees of freedom in the robot model.

        Returns
        -------
        List[int]
            A list of integer indices corresponding to the degrees of freedom in the robot model.
        """
        return self._dof_indxs

    @property
    def joint_ids(self):
        """
        List of joint IDs associated with the robot model.

        Returns
        -------
        List[int]
            A list of integer IDs representing the joints in the robot model.
        """
        return self._joint_ids

    @property
    def joint_names(self):
        """
        List of joint names associated with the robot model.

        Returns
        -------
        List[str]
            A list of names of the joints in the robot model.
        """
        return self._joint_names

    @property
    def n_actuators(self) -> int:
        """
        Get the number of actuators.

        Returns
        ----------
                Number of actuators.
        """
        return len(self.actuator_names)

    @property
    def n_joints(self) -> int:
        """
        Get the number of joints.

        Returns
        ----------
                Number of joints.
        """
        return len(self.joint_names)

    @property
    def joint_limits(self) -> np.ndarray:
        """
        Get the joint limits.

        Returns
        ----------
                Joint limits as a numpy array.
        """
        return np.array([get_joint_range(self._model, jn) for jn in self.joint_names]).T

    @property
    def actuator_limits(self) -> np.ndarray:
        """
        Get the actuator limits.

        Returns
        ----------
                Actuator limits as a numpy array.
        """
        return np.array(
            [get_actuator_range(self._model, an) for an in self.actuator_names]
        ).T


class JointType(Enum):
    """
    Enumeration of joint types used in MuJoCo simulations.

    This class defines the types of joints that can be used in a MuJoCo simulation,
    including free, ball, slide, and hinge joints. These joint types are associated
    with integer values that are used within the MuJoCo framework to specify the type
    of joint for different elements of the model.

    Attributes
    ----------
    FREE : int
        A joint that allows for unrestricted movement in all translational and rotational
        degrees of freedom. Represented by the integer value 0.
    BALL : int
        A ball-and-socket joint that allows rotation in all directions, but no translational
        movement. Represented by the integer value 1.
    SLIDE : int
        A prismatic joint that allows translational movement along a single axis, with no
        rotational freedom. Represented by the integer value 2.
    HINGE : int
        A rotational joint that allows rotation around a single axis, similar to a door hinge.
        Represented by the integer value 3.
    """

    FREE: int = 0
    BALL: int = 1
    SLIDE: int = 2
    HINGE: int = 3


def get_number_of_generalized_coordinates(model: mj.MjModel) -> int:
    """
    Retrieves the number of generalized coordinates (qpos) in the MuJoCo simulation.

    Generalized coordinates represent the state variables that define the configuration
    of the MuJoCo model. This function returns the total number of these coordinates,
    which is essential for understanding the model's degrees of freedom.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object from which to retrieve the number of generalized coordinates.

    Returns
    ----------
    int
        The number of generalized coordinates (nq) in the MuJoCo model.
    """
    return model.nq


def get_number_of_dof(model: mj.MjModel) -> int:
    """
    Retrieves the number of degrees of freedom (DOF) in the MuJoCo simulation.

    The degrees of freedom (DOF) represent the independent motions allowed
    in the MuJoCo model, such as translational and rotational movements.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of degrees of freedom (nv) in the MuJoCo model.
    """
    return model.nv


def get_number_of_actuators(model: mj.MjModel) -> int:
    """
    Retrieves the number of actuators in the MuJoCo simulation.

    Actuators in MuJoCo are responsible for generating forces or torques
    that drive the motion of the model's components.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of actuators (nu) in the MuJoCo model.
    """
    return model.nu


def get_number_of_activation_states(model: mj.MjModel) -> int:
    """
    Retrieves the number of activation states in the MuJoCo simulation.

    Activation states are used to model internal states of actuators,
    such as muscle activation levels or control signals.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of activation states (na) in the MuJoCo model.
    """
    return model.na


def get_number_of_bodies(model: mj.MjModel) -> int:
    """
    Retrieves the number of bodies in the MuJoCo simulation.

    Bodies are the rigid components that make up the physical structure
    of the model, such as links in a robotic arm.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of bodies (nbody) in the MuJoCo model.
    """
    return model.nbody


def get_number_of_joints(model: mj.MjModel) -> int:
    """
    Retrieves the number of joints in the MuJoCo simulation.

    Joints define the connections between bodies in the MuJoCo model,
    allowing for relative motion between them.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of joints (njnt) in the MuJoCo model.
    """
    return model.njnt


def get_number_of_geoms(model: mj.MjModel) -> int:
    """
    Retrieves the number of geometries (geoms) in the MuJoCo simulation.

    Geometries (geoms) are the shapes associated with bodies, defining
    their physical appearance and collision properties.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of geometries (ngeom) in the MuJoCo model.
    """
    return model.ngeom


def get_number_of_sites(model: mj.MjModel) -> int:
    """
    Retrieves the number of sites in the MuJoCo simulation.

    Sites are user-defined points of interest within the model, often
    used for attaching sensors, visual markers, or other features.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of sites (nsite) in the MuJoCo model.
    """
    return model.nsite


def get_number_of_cameras(model: mj.MjModel) -> int:
    """
    Retrieves the number of cameras in the MuJoCo simulation.

    Cameras in MuJoCo are used to render the simulation environment from
    specific viewpoints, either for visualization or for generating data.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of cameras (ncam) in the MuJoCo model.
    """
    return model.ncam


def get_number_of_equalities(model: mj.MjModel) -> int:
    """
    Retrieves the number of equality constraints in the MuJoCo simulation.

    Equality constraints are conditions enforced between model elements,
    such as keeping two points at a fixed distance or ensuring aligned rotations.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of equality constraints (neq) in the MuJoCo model.
    """
    return model.neq


def get_number_of_sensors(model: mj.MjModel) -> int:
    """
    Retrieves the number of sensors in the MuJoCo simulation.

    This function returns the total number of sensors defined in the MuJoCo model. Sensors
    provide feedback on various physical properties such as position, velocity, and force
    within the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object associated with the simulation.

    Returns
    ----------
    int
        The number of sensors (nsensor) in the MuJoCo model.
    """
    return model.nsensor


def get_number_of_keyframes(model: mj.MjModel) -> int:
    """
    Retrieves the number of keyframes defined in the MuJoCo model.

    Keyframes in MuJoCo capture the state of the simulation at specific points in time,
    including information such as position, velocity, and actuation. This function
    returns the total number of keyframes specified in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the keyframes.

    Returns
    ----------
    int
        The total number of keyframes (nkey) in the MuJoCo model.
    """
    return model.nkey


def get_actuator_names(model: mj.MjModel) -> List[str]:
    """
    Retrieves a list of actuator names from the MuJoCo model.

    This function returns the names of all actuators defined in the MuJoCo model.
    Actuators are components responsible for driving the motion of joints or other
    elements in the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object from which to retrieve actuator names.

    Returns
    ----------
    List[str]
        A list containing the names of all actuators in the MuJoCo model.
    """
    return [model.actuator(i).name for i in range(get_number_of_actuators(model))]


def get_actuator_ctrl(data: mj.MjData, model: mj.MjModel, actuator_name: str) -> float:
    """
    Retrieves the control value for a specific actuator in the MuJoCo simulation.

    The control value corresponds to the input command that drives the actuator,
    determining the force or torque applied in the simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object, which contains the state of the simulation.
    actuator_name : str
        The name of the actuator for which the control value is to be retrieved.

    Returns
    ----------
    float
        The control value of the specified actuator.
    """
    actuator_name = (
        actuator_name
        if isinstance(actuator_name, int)
        else actuator_name2id(model, actuator_name)
    )
    return data.actuator(actuator_name).ctrl[0]


def set_actuator_ctrl(
    data: mj.MjData,
    model: mj.MjModel,
    actuator_name: str,
    ctrl: Union[np.ndarray, float],
) -> None:
    """
    Sets the control value for a specific actuator in the MuJoCo simulation.

    This function allows you to set the input command for an actuator,
    which will influence the force or torque applied in the simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object, which contains the state of the simulation.
    actuator_name : str
        The name of the actuator for which the control value is to be set.
    ctrl : Union[np.ndarray, float]
        The control value(s) to be set for the actuator. This can be a single
        float or an array of values, depending on the actuator's configuration.

    Returns
    ----------
    None
    """
    actuator_name = (
        actuator_name
        if isinstance(actuator_name, int)
        else actuator_name2id(model, actuator_name)
    )
    data.actuator(actuator_name).ctrl = ctrl


def get_actuator_range(model: mj.MjModel, actuator_name: Union[int, str]) -> np.ndarray:
    """
    Retrieves the control range of a specific actuator in the MuJoCo model.

    The control range defines the minimum and maximum values that can be applied
    to the actuator during the simulation. This function supports retrieving
    the control range using either the actuator's index or its name.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object from which to retrieve the actuator's control range.
    actuator_name : Union[int, str]
        The identifier of the actuator for which the control range is to be retrieved.
        This can be either the actuator's index (int) or name (str).

    Returns
    ----------
    np.ndarray
        An array containing the control range [min, max] for the specified actuator.
    """
    return model.actuator(actuator_name).ctrlrange


def get_joint_names(model: mj.MjModel) -> List[str]:
    """
    Retrieves a list of joint names from the MuJoCo model.

    This function returns the names of all joints defined in the MuJoCo model.
    Joints connect different bodies and define the degrees of freedom for relative
    motion between them.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object from which to retrieve joint names.

    Returns
    ----------
    List[str]
        A list containing the names of all joints in the MuJoCo model.
    """
    return [model.joint(i).name for i in range(get_number_of_joints(model))]


def get_joint_range(model, joint_name: Union[int, str]) -> np.ndarray:
    """
    Retrieves the range of motion for a specific joint in the MuJoCo model.

    The range of motion defines the minimum and maximum allowable positions or
    angles for the joint during the simulation. This function supports retrieving
    the joint range using either the joint's index or its name.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object from which to retrieve the joint's range.
    joint_name : Union[int, str]
        The identifier of the joint for which the range is to be retrieved.
        This can be either the joint's index (int) or name (str).

    Returns
    ----------
    np.ndarray
        An array containing the range [min, max] for the specified joint.
    """

    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )

    return model.joint(joint_name).range


def set_freejoint_pose(
    data: mj.MjData, model: mj.MjModel, joint_name: Union[int, str], T: sm.SE3
) -> None:
    """
    Sets the pose of a free joint in the MuJoCo model.

    This function updates the position and orientation of a specified free joint
    in the MuJoCo simulation to match the given transformation matrix `T`. The
    transformation matrix is decomposed into its translational (t) and rotational
    (R) components, where the rotation is converted into a quaternion. These values
    are then used to update the corresponding joint's position and orientation in
    the simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object, which contains the state of the simulation.
    model : mj.MjModel
        The MuJoCo model object, representing the physical and geometric
        properties of the simulation.
    joint_name : Union[int, str]
        The identifier of the free joint whose pose is to be set. This can be either
        the joint's index (int) or name (str).
    T : sm.SE3
        The desired transformation matrix for the joint, including translational
        and rotational components.

    Returns
    ----------
    None

    Notes
    ----------
    Ensure that the specified `joint_name` corresponds to a free joint in the
    MuJoCo model. This function is specifically designed for joints with
    6 degrees of freedom (3 for translation and 3 for rotation).
    """
    t = T.t
    q = smb.r2q(T.R).data
    T_new = np.append(t, q)
    set_joint_q(data, model, joint_name, T_new)


def get_freejoint_pose(
    data: mj.MjData, model: mj.MjModel, joint_name: Union[int, str]
) -> sm.SE3:
    """
    Retrieves the pose of a free joint in the MuJoCo model.

    This function obtains the current position and orientation of a specified
    free joint in the MuJoCo simulation. The joint's state is extracted from
    the MuJoCo data object, with the position and orientation being converted
    into a transformation matrix (`SE3`). The orientation is represented as a
    rotation matrix, derived from the quaternion stored in the joint's state.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object, which contains the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object, representing the simulation's physical and
        geometric properties.
    joint_name : Union[int, str]
        The identifier of the free joint whose pose is to be retrieved. This can be
        either the joint's index (int) or name (str).

    Returns
    ----------
    sm.SE3
        The current transformation matrix of the joint, including both
        translational and rotational components.

    Notes
    ----------
    Ensure that the specified `joint_name` corresponds to a free joint in the
    MuJoCo model. This function is specifically designed for joints with
    6 degrees of freedom (3 for translation and 3 for rotation).
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )
    q_free = data.joint(joint_name).qpos
    T = make_tf(pos=q_free[:3], ori=q_free[3:])
    return T


def get_body_names(model: mj.MjModel) -> List[str]:
    """
    Retrieves the names of all bodies in the MuJoCo model.

    This function returns a list of names for all bodies defined in the provided
    MuJoCo model. Each body name corresponds to an entity in the simulation, such
    as a robot link or an object.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[str]
        A list of strings, each representing the name of a body in the MuJoCo model.
    """
    return [model.body(i).name for i in range(get_number_of_bodies(model))]


def get_body_ids(model: mj.MjModel) -> List[int]:
    """
    Retrieves the IDs of all bodies in the MuJoCo model.

    This function returns a list of IDs for all bodies defined in the provided
    MuJoCo model. These IDs uniquely identify each body within the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[int]
        A list of integers, each representing the ID of a body in the MuJoCo model.
    """
    return [model.body(i).id for i in range(get_number_of_bodies(model))]


def get_geom_names(model: mj.MjModel) -> List[str]:
    """
    Retrieves the names of all geometries (geoms) in the MuJoCo model.

    This function returns a list of names for all geometries defined in the provided
    MuJoCo model. Geometries include shapes and visual elements used in the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[str]
        A list of strings, each representing the name of a geometry in the MuJoCo model.
    """
    return [model.geom(i).name for i in range(get_number_of_geoms(model))]


def get_geom_ids(model: mj.MjModel) -> List[int]:
    """
    Retrieves the IDs of all geometries (geoms) in the MuJoCo model.

    This function returns a list of IDs for all geometries defined in the provided
    MuJoCo model. These IDs uniquely identify each geometry within the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[int]
        A list of integers, each representing the ID of a geometry in the MuJoCo model.
    """
    return [model.geom(i).id for i in range(get_number_of_geoms(model))]


def get_sensor_names(model: mj.MjModel) -> List[str]:
    """
    Retrieves the names of all sensors in the MuJoCo model.

    This function returns a list of names for all sensors defined in the provided
    MuJoCo model. Sensors gather various types of data from the simulation, such as
    position, velocity, and force information.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[str]
        A list of strings, each representing the name of a sensor in the MuJoCo model.
    """
    return [model.sensor(i).name for i in range(get_number_of_sensors(model))]


def get_sensor_ids(model: mj.MjModel) -> List[int]:
    """
    Retrieves the IDs of all sensors in the MuJoCo model.

    This function returns a list of IDs for all sensors defined in the provided
    MuJoCo model. These IDs uniquely identify each sensor within the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[int]
        A list of integers, each representing the ID of a sensor in the MuJoCo model.
    """
    return [model.sensor(i).id for i in range(get_number_of_sensors(model))]


def get_equality_names(model: mj.MjModel) -> List[str]:
    """
    Retrieves the names of all equality constraints in the MuJoCo model.

    This function returns a list of names for all equality constraints defined in the
    provided MuJoCo model. Each equality constraint enforces a specific relationship
    between two objects or elements in the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[str]
        A list of strings, each representing the name of an equality constraint in the MuJoCo model.
    """
    return [model.eq(i).name for i in range(get_number_of_equalities(model))]


def get_equality_ids(model: mj.MjModel) -> List[int]:
    """
    Retrieves the IDs of all equality constraints in the MuJoCo model.

    This function returns a list of IDs for all equality constraints defined in the
    provided MuJoCo model. These IDs uniquely identify each equality constraint within
    the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[int]
        A list of integers, each representing the ID of an equality constraint in the MuJoCo model.
    """
    return [model.eq(i).id for i in range(get_number_of_equalities(model))]


def get_keyframe_names(model: mj.MjModel) -> List[str]:
    """
    Retrieves the names of all keyframes in the MuJoCo model.

    This function returns a list of names for all keyframes defined in the provided
    MuJoCo model. Keyframes capture the state of the simulation at specific points
    in time, including information such as position, velocity, and actuation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the keyframe information.

    Returns
    ----------
    List[str]
        A list of strings, each representing the name of a keyframe in the MuJoCo model.
    """
    return [model.key(i).name for i in range(get_number_of_keyframes(model))]


def get_keyframe_ids(model: mj.MjModel) -> List[int]:
    """
    Retrieves the IDs of all keyframes in the MuJoCo model.

    This function returns a list of IDs for all keyframes defined in the provided
    MuJoCo model. These IDs uniquely identify each keyframe within the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the keyframe information.

    Returns
    ----------
    List[int]
        A list of integers, each representing the ID of a keyframe in the MuJoCo model.
    """
    return [model.key(i).id for i in range(get_number_of_keyframes(model))]


def get_body_pose(
    data: mj.MjData, model: mj.MjModel, body_name: Union[int, str]
) -> sm.SE3:
    """
    Retrieves the pose (position and orientation) of a specific body in the MuJoCo simulation.

    This function extracts the current position and orientation of a specified body
    from the MuJoCo data object. The pose is returned as a transformation matrix (`SE3`),
    combining translational and rotational components.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object, representing the simulation's physical and geometric
        properties.
    body_name : Union[int, str]
        The identifier of the body whose pose is to be retrieved. This can be either the
        body's index (int) or name (str).

    Returns
    ----------
    sm.SE3
        The current pose of the specified body, including both translational and rotational
        components.
    """
    body_name = (
        body_name if isinstance(body_name, int) else body_name2id(model, body_name)
    )
    t = data.body(body_name).xpos
    q = data.body(body_name).xquat
    return make_tf(pos=t, ori=q)


def get_geom_pose(
    data: mj.MjData, model: mj.MjModel, geom_name: Union[int, str]
) -> sm.SE3:
    """
    Retrieves the pose (position and orientation) of a specific geometry (geom) in the MuJoCo simulation.

    This function extracts the current position and orientation of a specified geometry from
    the MuJoCo data object. The pose is returned as a transformation matrix (`SE3`), where
    the orientation is represented as a rotation matrix.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object, representing the simulation's physical and geometric properties.
    geom_name : Union[int, str]
        The identifier of the geometry whose pose is to be retrieved. This can be either the
        geometry's index (int) or name (str).

    Returns
    ----------
    sm.SE3
        The current pose of the specified geometry, including both translational (position)
        and rotational (orientation) components. The rotation is represented as a 3x3 matrix.
    """
    geom_name = (
        geom_name if isinstance(geom_name, int) else geom_name2id(model, geom_name)
    )
    t = data.geom(geom_name).xpos
    R = np.array(data.geom(geom_name).xmat).reshape((3, 3))
    return make_tf(pos=t, ori=R)


def get_cam_pose(
    data: mj.MjData, model: mj.MjModel, cam_name: Union[str, int]
) -> sm.SE3:
    """
    Retrieves the pose (position and orientation) of a specific camera in the MuJoCo simulation.

    This function extracts the current position and orientation of a specified camera from
    the MuJoCo data object. The pose is returned as a transformation matrix (`SE3`), where
    the orientation is represented as a rotation matrix.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object that provides the mapping of camera names to indices.
    cam_name : Union[str, int]
        The identifier of the camera. This can be either the camera's name (str) or its index (int).

    Returns
    ----------
    sm.SE3
        The current pose of the specified camera, including both translational (position)
        and rotational (orientation) components. The rotation is represented as a 3x3 matrix.
    """
    cam_name = cam_name if isinstance(cam_name, int) else cam_name2id(model, cam_name)
    t = data.cam(cam_name).xpos
    R = np.array(data.cam(cam_name).xmat).reshape((3, 3))
    return make_tf(pos=t, ori=R)


def set_joint_qpos0(
    model: mj.MjModel, joint_name: Union[int, str], qpos0: Union[np.ndarray, float]
) -> None:
    """
    Sets the default position of a joint in the MuJoCo model.

    This function updates the default position (`qpos0`) of a specified joint within
    the MuJoCo model. The default position is used for initialization or resetting
    the joint to its starting configuration.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object where the joint's default position is to be set.
    joint_name : Union[int, str]
        The name or ID of the joint whose default position is being updated. Can be
        specified either as a string (name) or integer (ID).
    qpos0 : Union[np.ndarray, float]
        The default position value(s) to set for the joint. This can be a single float or
        an array of values, depending on the type of joint.

    Returns
    ----------
    None
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )

    model.joint(joint_name).qpos0 = qpos0


def get_joint_qpos_addr(model: mj.MjModel, joint_name: Union[int, str]) -> int:
    """
    Retrieves the address of the generalized position (qpos) for a specific joint in the MuJoCo model.

    This function returns the index of the joint's generalized position in the `qpos` array
    of the MuJoCo model. This index can be used to access or modify the joint's position
    directly in the simulation data.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object from which to retrieve the joint's qpos address.
    joint_name : Union[int, str]
        The name or ID of the joint whose qpos address is being retrieved. Can be specified
        either as a string (name) or an integer (ID).

    Returns
    ----------
    int
        The address (index) of the joint's generalized position in the `qpos` array.

    Notes
    -----
    Ensure that the specified `joint_name` corresponds to a valid joint in the MuJoCo model.
    """
    joint_id = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )

    return model.jnt_qposadr[joint_id]


def set_joint_q(
    data: mj.MjData,
    model: mj.MjModel,
    joint_name: Union[int, str],
    q: Union[np.ndarray, float],
    unit: str = "rad",
) -> None:
    """
    Sets the position(s) (angle(s)) of a joint in the MuJoCo simulation.

    This function updates the generalized position (`qpos`) of a specified joint in
    the MuJoCo simulation. The joint's position can be set in radians or degrees,
    depending on the `unit` parameter. If degrees are provided, they will be converted
    to radians before being applied. The positions are assigned to the joint's indices
    in the `qpos` array of the simulation data.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose position is being set.
    q : Union[np.ndarray, float]
        The position(s) (angle(s)) to be set for the joint. Can be a single value or an array
        of values, depending on the type of joint.
    unit : str, optional
        The unit of the position value, either "rad" (radians) or "deg" (degrees). Defaults to "rad".

    Returns
    ----------
    None

    Raises
    ------
    ValueError
        If the dimensions of `q` do not match the number of positions for the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - If the unit is "deg", the function will convert the angles from degrees to radians before applying.
    - Ensure that the `q` values match the dimensions expected for the joint.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )

    # Convert q to radians if the unit is degrees
    if unit == "deg":
        q = np.deg2rad(q)

    q_indxs = get_joint_qpos_indxs(data, model, joint_name)

    # Ensure q is a numpy array
    if isinstance(q, (int, float)):
        q = np.array([q])
    if isinstance(q, list):
        q = np.array(q)

    # Validate the dimensions of q
    if q.shape[0] != len(q_indxs):
        raise ValueError(
            f"Dimension mismatch: Expected dimension {len(q_indxs)}, "
            f"but got {q.shape[0]} for joint '{joint_name}'."
        )

    data.qpos[q_indxs] = q


def set_joint_dq(
    data: mj.MjData,
    model: mj.MjModel,
    joint_name: Union[int, str],
    dq: Union[np.ndarray, float],
    unit: str = "rad",
) -> None:
    """
    Sets the velocity of a joint in the MuJoCo simulation.

    This function updates the generalized velocity (`qvel`) of a specified joint
    in the MuJoCo simulation. The velocity can be specified in radians or degrees,
    and will be converted to radians if necessary. The velocities are assigned to the
    joint's degrees of freedom in the `qvel` array of the simulation data.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose velocity is being set.
    dq : Union[np.ndarray, float]
        The velocity value(s) to be set for the joint. This can be a single value or
        an array of values, depending on the type of joint.
    unit : str, optional
        The unit of the velocity value, either "rad" (radians) or "deg" (degrees). Defaults to "rad".

    Returns
    ----------
    None

    Raises
    ------
    ValueError
        If the dimensions of `dq` do not match the number of degrees of freedom for the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - If the unit is "deg", the function will convert the velocities from degrees to radians before applying.
    - Ensure that the `dq` values match the dimensions expected for the joint.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )

    # Convert dq to radians if the unit is degrees
    if unit == "deg":
        dq = np.deg2rad(dq)

    dq_indxs = get_joint_dof_indxs(model, joint_name)

    # Ensure dq is a numpy array
    if isinstance(dq, (int, float)):
        dq = np.array([dq])
    if isinstance(dq, list):
        dq = np.array(dq)

    # Validate the dimensions of dq
    if dq.shape[0] != len(dq_indxs):
        raise ValueError(
            f"Dimension mismatch: Expected dimension {len(dq_indxs)}, "
            f"but got {dq.shape[0]} for joint '{joint_name}'."
        )
    data.qvel[dq_indxs] = dq


def set_joint_ddq(
    data: mj.MjData,
    model: mj.MjModel,
    joint_name: Union[int, str],
    ddq: Union[np.ndarray, float],
    unit: str = "rad",
) -> None:
    """
    Sets the acceleration of a joint in the MuJoCo simulation.

    This function updates the generalized acceleration (`qacc`) of a specified joint
    in the MuJoCo simulation. The acceleration can be specified in radians or degrees,
    and will be converted to radians if necessary. The accelerations are assigned to the
    joint's degrees of freedom in the `qacc` array of the simulation data.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose acceleration is being set.
    ddq : Union[np.ndarray, float]
        The acceleration value(s) to be set for the joint. This can be a single value or
        an array of values, depending on the type of joint.
    unit : str, optional
        The unit of the acceleration value, either "rad" (radians) or "deg" (degrees). Defaults to "rad".

    Returns
    ----------
    None

    Raises
    ------
    ValueError
        If the dimensions of `ddq` do not match the number of degrees of freedom for the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - If the unit is "deg", the function will convert the accelerations from degrees to radians before applying.
    - Ensure that the `ddq` values match the dimensions expected for the joint.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )

    # Convert ddq to radians if the unit is degrees
    if unit == "deg":
        ddq = np.deg2rad(ddq)

    ddq_indxs = get_joint_dof_indxs(model, joint_name)

    # Ensure ddq is a numpy array
    if isinstance(ddq, (int, float)):
        ddq = np.array([ddq])
    if isinstance(ddq, list):
        ddq = np.array(ddq)

    # Validate the dimensions of ddq
    if ddq.shape[0] != len(ddq_indxs):
        raise ValueError(
            f"Dimension mismatch: Expected dimension {len(ddq_indxs)}, "
            f"but got {ddq.shape[0]} for joint '{joint_name}'."
        )
    data.qacc[ddq_indxs] = ddq


def get_joint_q(
    data: mj.MjData, model: mj.MjModel, joint_name: Union[int, str]
) -> np.ndarray:
    """
    Retrieves the position (angle) of a joint in the MuJoCo simulation.

    This function extracts the current position (or angle) of a specified joint
    from the MuJoCo simulation data. The joint position is obtained from the
    `qpos` array, which holds the generalized positions for all joints in the model.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose position is being retrieved.

    Returns
    ----------
    np.ndarray
        An array containing the position (or angle) of the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - The returned array includes the position values corresponding to the joint's degrees of freedom.
    - Ensure that the joint_name is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )
    q_indxs = get_joint_qpos_indxs(data, model, joint_name)
    return data.qpos[q_indxs]


def get_joint_dq(
    data: mj.MjData, model: mj.MjModel, joint_name: Union[int, str]
) -> np.ndarray:
    """
    Retrieves the velocity of a joint in the MuJoCo simulation.

    This function extracts the current velocity of a specified joint from the
    MuJoCo simulation data. The joint's velocity is obtained from the `qvel` array,
    which holds the generalized velocities for all joints in the model.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose velocity is being retrieved.

    Returns
    ----------
    np.ndarray
        An array containing the velocity values of the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - The returned array includes the velocity values corresponding to the joint's degrees of freedom.
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )
    dq_indxs = get_joint_dof_indxs(model, joint_name)
    return data.qvel[dq_indxs]


def get_joint_ddq(data: mj.MjData, model: mj.MjModel, joint_name: str) -> np.ndarray:
    """
    Retrieves the acceleration of a joint in the MuJoCo simulation.

    This function extracts the current acceleration of a specified joint from the
    MuJoCo simulation data. The joint's acceleration is obtained from the `qacc` array,
    which holds the generalized accelerations for all joints in the model.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose acceleration is being retrieved.

    Returns
    ----------
    np.ndarray
        An array containing the acceleration values of the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - The returned array includes the acceleration values corresponding to the joint's degrees of freedom.
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )
    ddq_indxs = get_joint_dof_indxs(model, joint_name)
    return data.qacc[ddq_indxs]


def get_joint_type(model: mj.MjModel, joint_name: Union[int, str]) -> JointType:
    """
    Retrieves the type of a joint in the MuJoCo simulation.

    This function returns the type of a specified joint from the MuJoCo model. The joint type
    is identified using its ID or name, and the function returns an enumerated type representing
    the joint's type.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose type is being retrieved.

    Returns
    ----------
    JointType
        An enumerated type representing the type of the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )
    return JointType(model.joint(joint_name).type[0])


def get_joint_ids(model: mj.MjModel) -> List[int]:
    """
    Retrieves the IDs of all joints in the MuJoCo simulation.

    This function returns a list of IDs for all joints present in the MuJoCo model. The IDs
    are obtained from the model and are used to identify joints uniquely within the simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.

    Returns
    ----------
    List[int]
        A list of integers representing the IDs of all joints in the simulation.

    Notes
    -----
    - The IDs are retrieved for all joints in the model based on the number of joints in the model.
    """
    return [model.joint(jid).id for jid in range(get_number_of_joints(model))]


def get_joint_qpos_indxs(
    data: mj.MjModel, model: mj.MjModel, joint_name: Union[int, str]
) -> np.ndarray:
    """
    Retrieves the indices in the `qpos` array corresponding to the specified joint in the MuJoCo model.

    This function determines the indices in the `qpos` array where the position(s) of the specified joint are stored.
    The indices are computed based on the joint's position address and its dimension.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose `qpos` indices are to be retrieved.

    Returns
    ----------
    np.ndarray
        An array of indices in the `qpos` array that correspond to the specified joint.

    Notes
    -----
    - The `joint_name` can be provided as either a string (name) or an integer (ID).
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )
    addr = get_joint_qpos_addr(model, joint_name)
    joint_dim = get_joint_dim(data, model, joint_name)
    return list(range(addr, addr + joint_dim))


def get_joint_dof_indxs(model: mj.MjModel, joint_name: Union[int, str]) -> np.ndarray:
    """
    Retrieves the degrees of freedom (DOF) indices for a specified joint in the MuJoCo model.

    This function obtains the indices of the degrees of freedom for a given joint, which represent
    the joint's DOF in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose DOF indices are to be retrieved.

    Returns
    ----------
    np.ndarray
        An array of indices corresponding to the degrees of freedom of the specified joint.

    Notes
    -----
    - The `joint_name` can be provided as either a string (name) or an integer (ID).
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )
    jdof = model.jnt_dofadr[joint_name]
    if not isinstance(jdof, np.ndarray):
        jdof = np.array([jdof])
    return jdof


def get_model_joint_dof_indxs(model: mj.MjModel, model_name: str) -> np.ndarray:
    """
    Retrieves the degrees of freedom (DOF) indices for all joints in a specified MuJoCo model.

    This function gathers the DOF indices for all joints that are part of a specified model name.
    It flattens the indices from each joint into a single array.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    model_name : str
        The name of the model for which the joint DOF indices are to be retrieved.

    Returns
    ----------
    np.ndarray
        A flattened array of DOF indices for all joints in the specified model.

    Notes
    -----
    - Ensure that `model_name` corresponds to an existing model in the MuJoCo system.
    """
    joint_ids = get_model_joint_ids(model, model_name)
    return np.array([get_joint_dof_indxs(model, jid) for jid in joint_ids]).flatten()


def get_joint_dim(
    data: mj.MjData, model: mj.MjModel, joint_name: Union[str, int]
) -> int:
    """
    Retrieves the dimensionality (number of `qpos` elements) of the specified joint in the MuJoCo simulation.

    This function determines the number of position elements (`qpos`) associated with a given joint,
    which reflects the joint's degrees of freedom or configuration.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[str, int]
        The name or ID of the joint whose dimensionality is to be retrieved.

    Returns
    ----------
    int
        The number of `qpos` elements for the specified joint, representing its dimensionality.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name if isinstance(joint_name, int) else joint_name2id(model, joint_name)
    )
    return len(data.joint(joint_name).qpos)


def get_mj_model(scene_path: str) -> mj.MjModel:
    """
    Load a MuJoCo model from an XML file or XML string.

    This function loads a MuJoCo model configuration from either a file path or an XML string,
    and returns the corresponding `mj.MjModel` object.

    Parameters
    ----------
    scene_path : str
        The file path to the XML scene file or the XML string containing the MuJoCo model.

    Returns
    ----------
    mj.MjModel
        The MuJoCo model object loaded from the XML file or string.

    Raises
    ------
    ValueError
        If the XML file or string is invalid or cannot be parsed.
    """
    try:
        # Try to load from file path
        if os.path.isfile(scene_path):
            model = mj.MjModel.from_xml_path(scene_path)
        else:
            raise FileNotFoundError("Path does not exist or is not a file.")
    except (FileNotFoundError, IOError):
        try:
            # Try to load from XML string
            model = mj.MjModel.from_xml_string(scene_path)
        except Exception as e:
            # Raise a ValueError if both attempts fail
            raise ValueError(
                f"Failed to load or parse the XML file or string. Error: {e}"
            )

    return model


def get_mj_data(model: mj.MjModel) -> mj.MjData:
    """
    Initialize a MuJoCo data structure from a given model.

    This function creates and returns a MuJoCo data structure (`mj.MjData`) initialized with the
    provided MuJoCo model. The data structure holds the simulation state and can be used to
    interact with or manipulate the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object for which to create the data structure.

    Returns
    ----------
    mj.MjData
        The MuJoCo data structure initialized from the given model.

    Raises
    ------
    ValueError
        If the provided model object is invalid or cannot be processed.
    """
    return mj.MjData(model)


def get_mj_camera() -> mj.MjvCamera:
    """
    Retrieves a MuJoCo camera instance.

    This function creates and returns an instance of the MuJoCo camera class, which can be used
    for setting up and managing the camera view within the MuJoCo simulation environment.

    Returns
    -------
    mj.MjvCamera
        An instance of the MuJoCo camera class.

    Notes
    -----
    - This function is intended for internal use within the simulation framework.
    """
    return mj.MjvCamera()


def get_mj_options() -> mj.MjvOption:
    """
    Retrieves MuJoCo viewer options.

    This function creates and returns an instance of the MuJoCo viewer options class, which
    provides various configuration settings for rendering the simulation scene.

    Returns
    -------
    mj.MjvOption
        An instance of the MuJoCo viewer options class.

    Notes
    -----
    - This function is intended for internal use within the simulation framework.
    """
    return mj.MjvOption()


def get_mj_window(model: mj.MjModel, data: mj.MjData, keyboard_callback) -> Handle:
    """
    Initializes and retrieves a MuJoCo viewer window.

    This function launches a MuJoCo viewer window for interactive visualization of the simulation.
    It allows the user to visualize the simulation and interact with it via keyboard callbacks.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object to be visualized.
    data : mj.MjData
        The MuJoCo data object containing the simulation state.
    keyboard_callback
        A callback function to handle keyboard input.

    Returns
    -------
    Handle
        A handle to the MuJoCo viewer window.

    Notes
    -----
    - This function is intended for internal use within the simulation framework.
    """
    return mj.viewer.launch_passive(model, data, key_callback=keyboard_callback)


def get_mj_scene(model: mj.MjModel, maxgeom: int = 10000) -> mj.MjvScene:
    """
    Retrieves a MuJoCo scene instance.

    This function creates and returns an instance of the MuJoCo scene class, which is used for
    managing the scene elements and rendering parameters within the MuJoCo simulation environment.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object for the scene.
    maxgeom : int, optional
        The maximum number of geometries to be handled by the scene. Default is 10000.

    Returns
    -------
    mj.MjvScene
        An instance of the MuJoCo scene class.

    Notes
    -----
    - This function is intended for internal use within the simulation framework.
    """
    return mj.MjvScene(model, maxgeom=maxgeom)


def get_mj_perturbation() -> mj.MjvPerturb:
    """
    Retrieves a MuJoCo perturbation instance.

    This function creates and returns an instance of the MuJoCo perturbation class, which can
    be used for applying and managing perturbations (forces, torques, etc.) in the simulation.

    Returns
    -------
    mj.MjvPerturb
        An instance of the MuJoCo perturbation class.

    Notes
    -----
    - This function is intended for internal use within the simulation framework.
    """
    return mj.MjvPerturb()


def get_mj_context(
    model: mj.MjModel = None, font_code: int = mj.mjtFontScale.mjFONTSCALE_150
) -> mj.MjrContext:
    """
    Retrieves a MuJoCo context instance.

    This function creates and returns an instance of the MuJoCo context class, which manages
    the rendering context and provides options for text rendering within the MuJoCo simulation.

    Parameters
    ----------
    model : mj.MjModel, optional
        The MuJoCo model object for the context. If not provided, a default context is created.
    font_code : int, optional
        The font scale code for rendering text. Default is `mj.mjtFontScale.mjFONTSCALE_150`.

    Returns
    -------
    mj.MjrContext
        An instance of the MuJoCo context class.

    Notes
    -----
    - This function is intended for internal use within the simulation framework.
    """
    if model is not None:
        return mj.MjrContext(model, font_code)
    return mj.MjrContext()


def get_sensor_data(
    data: mj.MjData, model: mj.MjModel, sensor_name: Union[str, int]
) -> np.ndarray:
    """
    Retrieves the data from a specified sensor in the MuJoCo simulation.

    This function extracts the data recorded by a sensor during the simulation. The sensor data
    is accessed from the `sensor` array within the MuJoCo data object, which contains measurements
    from various sensors defined in the simulation model.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object that contains the current simulation state and sensor readings.
    model : mj.MjModel
        The MuJoCo model object that defines the simulation environment and sensors.
    sensor_name : Union[str, int]
        The name or ID of the sensor whose data is to be retrieved. If a string is provided, it is
        converted to an ID using the `sensor_name2id` function. If an integer is provided, it is
        used directly as the sensor ID.

    Returns
    -------
    np.ndarray
        An array containing the data recorded by the specified sensor. The structure and length
        of this array depend on the type and configuration of the sensor.

    Notes
    -----
    - Ensure that the `sensor_name` or ID corresponds to an existing sensor in the MuJoCo model.
    - The returned data array may vary in size depending on the sensor's data type (e.g., position,
      velocity, force) and configuration.
    """
    sensor_name = (
        sensor_name
        if isinstance(sensor_name, int)
        else sensor_name2id(model, sensor_name)
    )
    return data.sensor(sensor_name).data


def sensor_name2id(model: mj.MjModel, sensor_name: str):
    """
    Converts a sensor name to its corresponding ID in the MuJoCo model.

    This function retrieves the unique identifier (ID) for a sensor based on its name. The ID is
    used to reference the sensor in MuJoCo's data and model structures.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object that contains the definition of all sensors.
    sensor_name : str
        The name of the sensor for which the ID is to be retrieved.

    Returns
    -------
    int
        The unique identifier of the sensor in the MuJoCo model.

    Raises
    ------
    ValueError
        If the sensor name does not correspond to any sensor in the MuJoCo model.

    Notes
    -----
    - This function uses MuJoCo's internal `mj_name2id` function to map the sensor name to its ID.
    - Ensure that the sensor name is correct and exists in the provided model.
    """
    # return mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, sensor_name)
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, sensor_name)
    if body_id == -1:
        raise ValueError(
            f"Body name '{sensor_name}' not found in the model. The model contain the geometries {get_sensor_names(model)}"
        )
    return body_id


def get_body_parent_id(model: mj.MjModel, body_name: Union[int, str]) -> int:
    """
    Retrieves the ID of the parent body of a specified body in the MuJoCo model.

    This function returns the unique identifier (ID) of the parent body for a given body.
    The parent ID is used to understand the hierarchical relationships between bodies in the model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object that contains the definition of all bodies.
    body_name : Union[int, str]
        The name or ID of the body whose parent's ID is to be retrieved.

    Returns
    -------
    int
        The unique identifier of the parent body.

    Notes
    -----
    - If `body_name` is provided as a name (string), it is converted to an ID using `body_name2id`.
    - Ensure that the body name or ID is correct and corresponds to an existing body in the model.
    """
    body_name = (
        body_name if isinstance(body_name, int) else body_name2id(model, body_name)
    )
    return model.body(body_name).parentid[0]


def get_body_parent_name(model: mj.MjModel, body_name: Union[int, str]) -> str:
    """
    Retrieves the name of the parent body of a specified body in the MuJoCo model.

    This function returns the name of the parent body for a given body.
    The parent body name helps to understand the hierarchical structure of bodies in the model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object that contains the definition of all bodies.
    body_name : Union[int, str]
        The name or ID of the body whose parent's name is to be retrieved.

    Returns
    -------
    str
        The name of the parent body.

    Notes
    -----
    - If `body_name` is provided as a name (string), it is converted to an ID using `body_name2id`.
    - Ensure that the body name or ID is correct and corresponds to an existing body in the model.
    """
    body_name = (
        body_name if isinstance(body_name, int) else body_name2id(model, body_name)
    )
    parent_id = model.body(body_name).parentid[0]
    return body_id2name(model, parent_id)


def body_id2name(model: mj.MjModel, body_id: int) -> str:
    """
    Convert a body ID to its corresponding name in the MuJoCo model.

    This function retrieves the name of a body based on its unique identifier (ID) within
    the MuJoCo model. The ID is used to look up the name in the model's data structure.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object that contains the body definitions.
    body_id : int
        The unique identifier of the body.

    Returns
    -------
    str
        The name of the body corresponding to the given ID.

    Notes
    -----
    - Ensure that the provided ID is valid and corresponds to an existing body in the model.
    """
    # return mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
    body_ids = get_body_ids(model)
    if body_id not in body_ids:
        raise ValueError(
            f"Body id '{body_id}' not found in the model. The model contain the bodyetries {body_ids}"
        )
    return body_name


def body_name2id(model: mj.MjModel, body_name: str) -> int:
    """
    Convert a body name to its corresponding ID in the MuJoCo model.

    This function retrieves the unique identifier (ID) of a body based on its name within
    the MuJoCo model. The name is used to look up the ID in the model's data structure.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object that contains the body definitions.
    body_name : str
        The name of the body.

    Returns
    -------
    int
        The unique identifier of the body corresponding to the given name.

    Notes
    -----
    - Ensure that the provided name is valid and corresponds to an existing body in the model.
    """
    # return mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(
            f"Body name '{body_name}' not found in the model. The model contain the geometries {get_body_names(model)}"
        )
    return body_id


def geom_id2name(model: mj.MjModel, geom_id: int) -> str:
    """
    Convert a geom ID to its corresponding name in the MuJoCo model.

    This function retrieves the name of a geom based on its unique identifier (ID) within
    the MuJoCo model. The ID is used to look up the name in the model's data structure.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object that contains the geom definitions.
    geom_id : int
        The unique identifier of the geom.

    Returns
    -------
    str
        The name of the geom corresponding to the given ID.

    Notes
    -----
    - Ensure that the provided ID is valid and corresponds to an existing geom in the model.
    """
    # return mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
    geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
    geom_ids = get_geom_ids(model)
    if geom_id not in geom_ids:
        raise ValueError(
            f"Geom id '{geom_id}' not found in the model. The model contain the geometries {geom_ids}"
        )
    return geom_name


def geom_name2id(model: mj.MjModel, geom_name: str) -> int:
    """
    Convert a geom name to its corresponding ID in the MuJoCo model.

    This function retrieves the unique identifier (ID) of a geom based on its name within
    the MuJoCo model. The name is used to look up the ID in the model's data structure.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object that contains the geom definitions.
    geom_name : str
        The name of the geom.

    Returns
    -------
    int
        The unique identifier of the geom corresponding to the given name.

    Notes
    -----
    - Ensure that the provided name is valid and corresponds to an existing geom in the model.
    - If the name does not exist in the model, this function will return an ID of `-1`.
    """
    geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id == -1:
        raise ValueError(
            f"Geom name '{geom_name}' not found in the model. The model contain the geometries {get_geom_names(model)}"
        )
    return geom_id


def eq_id2name(model: mj.MjModel, equality_id: int) -> str:
    """
    Convert an equality ID to its corresponding name in the MuJoCo model.

    This function retrieves the name of an equality constraint based on its unique identifier (ID) within
    the MuJoCo model. The ID is used to look up the name in the model's data structure.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object that contains the equality constraints.
    equality_id : int
        The unique identifier of the equality constraint.

    Returns
    -------
    str
        The name of the equality constraint corresponding to the given ID.

    Notes
    -----
    - Ensure that the provided ID is valid and corresponds to an existing equality constraint in the model.
    """
    # return mj.mj_id2name(model, mj.mjtObj.mjOBJ_EQUALITY, equality_id)
    eq_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_EQUALITY, equality_id)
    eq_ids = get_equality_ids(model)
    if equality_id not in eq_ids:
        raise ValueError(
            f"Equality id '{equality_id}' not found in the model. The model contain the equalities {eq_ids}"
        )
    return eq_name


def eq_name2id(model: mj.MjModel, equality_name: str) -> int:
    """
    Convert an equality name to its corresponding ID in the MuJoCo model.

    This function retrieves the unique identifier (ID) of an equality constraint based on its name within
    the MuJoCo model. The name is used to look up the ID in the model's data structure.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object that contains the equality constraints.
    equality_name : str
        The name of the equality constraint.

    Returns
    -------
    int
        The unique identifier of the equality constraint corresponding to the given name.

    Notes
    -----
    - Ensure that the provided name is valid and corresponds to an existing equality constraint in the model.
    """
    # return mj.mj_name2id(model, mj.mjtObj.mjOBJ_EQUALITY, equality_name)
    eq_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_EQUALITY, equality_name)
    if eq_id == -1:
        raise ValueError(
            f"Equality name '{equality_name}' not found in the model. The model contain the equalities {get_equality_names(model)}"
        )
    return eq_id


def get_cam_ids(model: mj.MjModel) -> List[int]:
    """
    Retrieves the IDs of all cameras in the MuJoCo model.

    This function returns a list of IDs for all cameras defined in the provided
    MuJoCo model. These IDs uniquely identify each camera within the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[int]
        A list of integers, each representing the ID of a camera in the MuJoCo model.
    """
    return [model.cam(i).id for i in range(get_number_of_cameras(model))]


def get_cam_names(model: mj.MjModel) -> List[str]:
    """
    Retrieves the names of all cameras in the MuJoCo model.

    This function returns a list of names for all cameras defined in the provided
    MuJoCo model. Cameras are used to capture different views of the simulation
    environment.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[str]
        A list of strings, each representing the name of a camera in the MuJoCo model.
    """
    return [model.cam(i).name for i in range(get_number_of_cameras(model))]


def cam_id2name(model: mj.MjModel, cam_id: int) -> str:
    """
    Convert a camera ID to its corresponding name in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the camera definitions.
    cam_id : int
        The ID of the camera.

    Returns
    -------
    str
        The name of the camera corresponding to the given ID.

    Notes
    -----
    - Ensure that the provided camera ID is valid and corresponds to an existing camera in the model.
    """
    # return mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, cam_id)
    cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, cam_id)
    cam_ids = get_cam_ids(model)
    if cam_id not in cam_ids:
        raise ValueError(
            f"Camera id '{cam_id}' not found in the model. The model contain the cameras {cam_ids}"
        )
    return cam_name


def cam_name2id(model: mj.MjModel, cam_name: str) -> int:
    """
    Convert a camera name to its corresponding ID in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the camera definitions.
    name : str
        The name of the camera.

    Returns
    -------
    int
        The ID of the camera corresponding to the given name.

    Notes
    -----
    - Ensure that the provided camera name is valid and corresponds to an existing camera in the model.
    """
    # return mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, name)
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        raise ValueError(
            f"Camera name '{cam_name}' not found in the model. The model contain the cameras {get_cam_names(model)}"
        )
    return cam_id


def get_site_ids(model: mj.MjModel) -> List[int]:
    """
    Retrieves the IDs of all sites in the MuJoCo model.

    This function returns a list of IDs for all sites defined in the provided
    MuJoCo model. Sites are often used as reference points or markers within the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[int]
        A list of integers, each representing the ID of a site in the MuJoCo model.
    """
    return [model.site(i).id for i in range(get_number_of_sites(model))]


def get_site_names(model: mj.MjModel) -> List[str]:
    """
    Retrieves the names of all sites in the MuJoCo model.

    This function returns a list of names for all sites defined in the provided
    MuJoCo model. Sites are typically used for defining locations or points of interest
    in the simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[str]
        A list of strings, each representing the name of a site in the MuJoCo model.
    """
    return [model.site(i).name for i in range(get_number_of_sites(model))]


def site_id2name(model: mj.MjModel, site_id: int) -> str:
    """
    Convert a site ID to its corresponding name in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the site definitions.
    id : int
        The ID of the site.

    Returns
    -------
    str
        The name of the site corresponding to the given ID.

    Notes
    -----
    - Ensure that the provided site ID is valid and corresponds to an existing site in the model.
    """
    # return mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, id)
    site_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, site_id)
    site_ids = get_site_ids(model)
    if site_id not in site_ids:
        raise ValueError(
            f"Site id '{site_id}' not found in the model. The model contain the sites {site_ids}"
        )
    return site_name


def site_name2id(model: mj.MjModel, site_name: str) -> int:
    """
    Convert a site name to its corresponding ID in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the site definitions.
    name : str
        The name of the site.

    Returns
    -------
    int
        The ID of the site corresponding to the given name.

    Notes
    -----
    - Ensure that the provided site name is valid and corresponds to an existing site in the model.
    """
    # return mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(
            f"Site name '{site_name}' not found in the model. The model contain the sites {get_site_names(model)}"
        )
    return site_id


def joint_id2name(model: mj.MjModel, joint_id: int) -> str:
    """
    Convert a joint ID to its corresponding name in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the joint definitions.
    id : int
        The ID of the joint.

    Returns
    -------
    str
        The name of the joint corresponding to the given ID.

    Notes
    -----
    - Ensure that the provided joint ID is valid and corresponds to an existing joint in the model.
    """
    # return mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, id)
    joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
    joint_ids = get_joint_ids(model)
    if joint_id not in joint_ids:
        raise ValueError(
            f"Joint id '{joint_id}' not found in the model. The model contain the joints {joint_ids}"
        )
    return joint_name


def joint_name2id(model: mj.MjModel, joint_name: str) -> int:
    """
    Convert a joint name to its corresponding ID in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the joint definitions.
    name : str
        The name of the joint.

    Returns
    -------
    int
        The ID of the joint corresponding to the given name.

    Notes
    -----
    - Ensure that the provided joint name is valid and corresponds to an existing joint in the model.
    """
    # return mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        raise ValueError(
            f"Joint name '{joint_name}' not found in the model. The model contain the joints {get_joint_names(model)}"
        )
    return joint_id


def keyframe_id2name(model: mj.MjModel, keyframe_id: int) -> str:
    """
    Convert a keyframe ID to its corresponding name in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the keyframe definitions.
    keyframe_id : int
        The ID of the keyframe.

    Returns
    -------
    str
        The name of the keyframe corresponding to the given ID.

    Notes
    -----
    - Ensure that the provided keyframe ID is valid and corresponds to an existing keyframe in the model.
    """
    keyframe_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_KEY, keyframe_id)
    key_frame_ids = get_keyframe_ids(model)
    if keyframe_id not in key_frame_ids:
        raise ValueError(
            f"Keyframe ID '{keyframe_id}' not found in the model. The model contains the keyframes {key_frame_ids}"
        )
    return keyframe_name


def keyframe_name2id(model: mj.MjModel, keyframe_name: str) -> int:
    """
    Convert a keyframe name to its corresponding ID in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the keyframe definitions.
    keyframe_name : str
        The name of the keyframe.

    Returns
    -------
    int
        The ID of the keyframe corresponding to the given name.

    Notes
    -----
    - Ensure that the provided keyframe name is valid and corresponds to an existing keyframe in the model.
    """
    keyframe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, keyframe_name)
    if keyframe_id == -1:
        raise ValueError(
            f"Keyframe name '{keyframe_name}' not found in the model. The model contains the keyframes {get_keyframe_names(model)}"
        )
    return keyframe_id


def get_actuator_ids(model: mj.MjModel) -> List[int]:
    """
    Retrieves the IDs of all actuators in the MuJoCo model.

    This function returns a list of IDs for all actuators defined in the provided
    MuJoCo model. Actuators are used to apply forces or torques to the simulation,
    controlling the movement of bodies and joints.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation's structure and properties.

    Returns
    ----------
    List[int]
        A list of integers, each representing the ID of an actuator in the MuJoCo model.
    """
    return [model.actuator(i).id for i in range(get_number_of_actuators(model))]


def actuator_id2name(model: mj.MjModel, actuator_id: int) -> str:
    """
    Convert an actuator ID to its corresponding name in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the actuator definitions.
    id : int
        The ID of the actuator.

    Returns
    -------
    str
        The name of the actuator corresponding to the given ID.

    Notes
    -----
    - Ensure that the provided actuator ID is valid and corresponds to an existing actuator in the model.
    """
    # return mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, id)
    actuator_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    actuator_ids = get_actuator_ids(model)
    if actuator_id not in actuator_ids:
        raise ValueError(
            f"Actuator id '{actuator_id}' not found in the model. The model contain the actuators {actuator_ids}"
        )
    return actuator_name


def actuator_name2id(model: mj.MjModel, actuator_name: str) -> int:
    """
    Convert an actuator name to its corresponding ID in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the actuator definitions.
    name : str
        The name of the actuator.

    Returns
    -------
    int
        The ID of the actuator corresponding to the given name.

    Notes
    -----
    - Ensure that the provided actuator name is valid and corresponds to an existing actuator in the model.
    """
    # return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
    actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id == -1:
        raise ValueError(
            f"Actuator name '{actuator_name}' not found in the model. The model contain the actuators {get_joint_names(model)}"
        )
    return actuator_id


def load_mj_state(
    model: mj.MjModel,
    data: mj.MjData,
    state_name: str,
    lock: Lock,
    return_xml: bool = False,
    file_path: Optional[str] = None,
    log: bool = True,
    step: bool = True,
) -> Optional[str]:
    """
    Load a MuJoCo simulation state from a specified keyframe in an XML file or string.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object which defines the simulation.
    data : mj.MjData
        The MuJoCo data object to load the state into.
    state_name : str
        The name of the keyframe state to load.
    lock : Lock
        A threading lock to ensure thread safety during state loading.
    return_xml : bool, optional
        If True, return the XML content as a string instead of loading it into the model. Defaults to False.
    file_path : str, optional
        Path to the XML file to load from. If None, defaults to "states/states.xml".
    log : bool, optional
        Whether to log the loading action. Defaults to True.
    step : bool, optional
        Whether to step the simulation after loading the state. Defaults to True.

    Returns
    -------
    Optional[str]
        If return_xml is True, returns the XML content as a string. Otherwise, returns None.

    Raises
    ------
    ValueError
        If the specified state_name does not correspond to a valid keyframe.
    FileNotFoundError
        If the specified file_path does not exist.
    """
    if file_path is None:
        file_path = "states/states.xml"

    with lock:
        if return_xml:
            # Load XML content from file and return as string
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    xml_content = file.read()
                return xml_content
            else:
                print(f"File {file_path} not found.")
                return None

        # Proceed with loading state from file
        if os.path.exists(file_path):
            tree = ET.parse(file_path)
            root = tree.getroot()

            keyframe_element = root.find(".//keyframe")
            if keyframe_element is not None:
                try:
                    state_id = model.keyframe(state_name).id
                except Exception as e:
                    raise ValueError(
                        f'"{state_name}" does not seem to be available as a state, have you remembered to include the keyframes in your scene file? ERROR({e})'
                    )

                mj.mj_resetDataKeyframe(model, data, state_id)
                if log:
                    print(f'Loaded : "{state_name}" keyframe')
        else:
            print(f"File {file_path} not found.")
        if step:
            mj.mj_step(model, data)


def save_mj_state(
    data: mj.MjData,
    state_name: str,
    save_path: Union[str, None] = "states/states.xml",
    xml_string: Optional[str] = None,
) -> None:
    """
    Save the current state of a MuJoCo simulation to an XML file or string.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current simulation state.
    state_name : str
        The name to associate with the saved state.
    save_path : str, optional
        The file path where the XML file will be saved. Defaults to "states/states.xml". If xml_string is provided, this is ignored.
    xml_string : str, optional
        If provided, saves the state as an XML string instead of a file. Defaults to None.

    Returns
    -------
    None
        The function does not return any value.

    Notes
    -----
    - If xml_string is provided, the save_path is ignored, and the state is saved directly as an XML string.
    - If the XML file already exists, the state with the given name will be updated if it exists, or a new entry will be created.
    - Illegal XML characters in the state data are replaced with appropriate representations to ensure XML validity.
    """
    import xml.etree.ElementTree as ET

    def replace_illegal_chars(text: str) -> str:
        """
        Replace illegal characters in XML text with appropriate representations.

        Parameters
        ----------
        text : str
            The text containing potential illegal XML characters.

        Returns
        -------
        str
            The text with illegal characters replaced.
        """
        return text.replace("&#10;", "")

    if xml_string is not None:
        # Save state as XML string
        xml_str = minidom.parseString(xml_string).toprettyxml(indent="    ")
        print(f"State {state_name} saved as XML string.")
        return

    # Check if the file exists or is empty
    if not os.path.exists(save_path) or os.stat(save_path).st_size == 0:
        # Create a new XML file with a root element
        root = ET.Element("mujocoinclude")
        keyframe_element = ET.SubElement(root, "keyframe")

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        tree = ET.parse(save_path)
        root = tree.getroot()

        # Find or create the keyframe element
        keyframe_elements = root.findall(".//keyframe")
        if not keyframe_elements:
            keyframe_element = ET.SubElement(root, "keyframe")
        else:
            keyframe_element = keyframe_elements[0]

        # Check if the state name already exists and update it
        existing_keys = keyframe_element.findall(f"./key[@name='{state_name}']")
        if existing_keys:
            existing_key = existing_keys[0]
            existing_key.set("time", replace_illegal_chars(str(data.time)))
            existing_key.set("qpos", replace_illegal_chars(str(data.qpos)[1:-1]))
            existing_key.set("qvel", replace_illegal_chars(str(data.qvel)[1:-1]))
            existing_key.set("ctrl", replace_illegal_chars(str(data.ctrl)[1:-1]))
            existing_key.set("mpos", replace_illegal_chars(str(data.mocap_pos)[2:-2]))
            existing_key.set("mquat", replace_illegal_chars(str(data.mocap_quat)[2:-2]))
            tree.write(save_path)
            print(f"Overwrote state {state_name} to {save_path}")
            return

    # Create a new key element
    new_key = ET.Element("key")
    new_key.set("name", state_name)
    new_key.set("time", replace_illegal_chars(str(data.time)))
    new_key.set("qpos", replace_illegal_chars(str(data.qpos)[1:-1]))
    new_key.set("qvel", replace_illegal_chars(str(data.qvel)[1:-1]))
    new_key.set("ctrl", replace_illegal_chars(str(data.ctrl)[1:-1]))
    new_key.set("mpos", replace_illegal_chars(str(data.mocap_pos)[2:-2]))
    new_key.set("mquat", replace_illegal_chars(str(data.mocap_quat)[2:-2]))

    # Append the new key to the keyframe element
    keyframe_element.append(new_key)

    # Save the XML file with pretty formatting
    xml_str = ET.tostring(root, encoding="utf-16")
    parsed_xml = minidom.parseString(xml_str)
    with open(save_path, "w") as file:
        file.write(parsed_xml.toprettyxml(indent="    "))

    print(f"Saved state {state_name} to {save_path}")


def get_model_body_ids(model: mj.MjModel, model_name: str) -> List[int]:
    """
    Retrieve the IDs of the bodies belonging to a specified model in a MuJoCo simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    model_name : str
        The name of the model whose body IDs are to be retrieved.

    Returns
    -------
    List[int]
        A list of integers representing the body IDs belonging to the specified model.

    Notes
    -----
    - This function constructs a tree structure of the simulation's bodies starting from the root ("world") and traverses it to find the bodies associated with the specified model.
    - Bodies are identified by their parent-child relationships, and only bodies not associated with the "world" are considered.
    """
    # Get all body names that are not "world"
    body_names = [bn for bn in get_body_names(model) if "world" not in bn]

    # Function to recursively get children of a body
    def get_children(parent_id: int, body_names: List[str], visited: set) -> List[int]:
        children = [
            i
            for i, name in enumerate(body_names)
            if model.body_parentid[i] == parent_id and i not in visited
        ]
        visited.update(children)
        for child in children:
            children.extend(get_children(child, body_names, visited))
        return children

    # Function to recursively get descendants of a body
    def get_descendants(body_id: int, body_names: List[str], visited: set) -> List[int]:
        descendants = [body_id]
        children = get_children(body_id, body_names, visited)
        for child in children:
            descendants.extend(get_descendants(child, body_names, visited))
        return descendants

    # Get IDs of children of "world" body
    children_of_world = [
        i for i, name in enumerate(body_names) if model.body_parentid[i] == 0
    ]

    # Generate tree structures for each child of "world"
    trees = [[child_id] for child_id in children_of_world]

    visited = set(children_of_world)

    for i, tree in enumerate(trees):
        children = get_children(tree[-1], body_names, visited)
        for child in children:
            trees[i].extend(get_descendants(child, body_names, visited))

    # Find the tree representing the model_name
    for tree in trees:
        for body_id in tree:
            if model_name in body_id2name(model, body_id):
                return tree

    # If model_name not found, return an empty list
    return []


def get_model_geom_ids(model: mj.MjModel, model_name: str) -> List[int]:
    """
    Retrieve the IDs of the geometries associated with a specified model in a MuJoCo simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    model_name : str
        The name of the model whose geometry IDs are to be retrieved.

    Returns
    -------
    List[int]
        A list of integers representing the geometry IDs associated with the specified model.

    Notes
    -----
    - Geometries are linked to bodies in the simulation, so the function first retrieves the body IDs associated with the specified model and then finds the geometries linked to those bodies.
    - The function iterates over all geometries in the model to match them with the relevant body IDs.
    """
    # Get body IDs belonging to the specified model
    body_ids = get_model_body_ids(model, model_name)

    # List to store geometry IDs
    geom_ids = []

    # Iterate through all geometries in the model
    for geom_id in range(model.ngeom):
        # Check if the geometry's bodyid matches any of the body IDs
        if model.geom_bodyid[geom_id] in body_ids:
            # If matched, add the geometry ID to the list
            geom_ids.append(geom_id)

    return geom_ids


def get_model_joint_ids(
    model: mj.MjModel, model_name: str, filter_freejoints: bool = True
) -> List[int]:
    """
    Retrieve the IDs of the joints associated with a specified model in a MuJoCo simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    model_name : str
        The name of the model whose joint IDs are to be retrieved.
    filter_freejoints : bool, optional
        Whether to filter out joints of type "free". Defaults to True.

    Returns
    -------
    List[int]
        A list of integers representing the joint IDs associated with the specified model.

    Notes
    -----
    - Joints are linked to bodies in the simulation, so the function first retrieves the body IDs associated with the specified model and then finds the joints linked to those bodies.
    - The `filter_freejoints` parameter allows for excluding joints of type "free", which are typically used for free-floating bodies.
    """
    # Get body IDs belonging to the specified model
    body_ids = get_model_body_ids(model, model_name)

    # List to store joint IDs
    joint_ids = []

    # Iterate through all joints in the model
    for jnt_id in range(model.njnt):
        # Check if the joint's bodyid matches any of the body IDs
        if model.jnt_bodyid[jnt_id] in body_ids:
            # If matched, add the joint ID to the list
            joint_ids.append(jnt_id)

    if filter_freejoints:
        joint_ids = [
            jid
            for jid in joint_ids
            if JointType(model.joint(jid).type[0]) is not JointType.FREE
        ]
    return joint_ids


def get_model_joint_qpos_indxs(
    data: mj.MjData, model: mj.MjModel, model_name: str, filter_freejoints: bool = True
) -> List[int]:
    """
    Retrieve the qpos indices of the joints associated with a specified model in a MuJoCo simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the simulation state.
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    model_name : str
        The name of the model whose joint qpos indices are to be retrieved.
    filter_freejoints : bool, optional
        Whether to filter out joints of type "free". Defaults to True.

    Returns
    -------
    List[int]
        A list of integers representing the qpos indices associated with the joints of the specified model.

    Notes
    -----
    - Joints are linked to bodies in the simulation, and the function first retrieves the joint IDs associated with the specified model before mapping them to their corresponding qpos indices.
    - The `filter_freejoints` parameter allows for excluding joints of type "free", which are typically used for free-floating bodies.
    - The qpos indices provide a mapping to the generalized positions in the state vector (`qpos`) that correspond to the specified joints.
    """
    # Get body IDs belonging to the specified model
    joint_ids = get_model_joint_ids(model, model_name)

    if filter_freejoints:
        joint_ids = [
            jid
            for jid in joint_ids
            if JointType(model.joint(jid).type[0]) is not JointType.FREE
        ]
    model_joint_indxs = np.array(
        [get_joint_qpos_indxs(data, model, jid) for jid in joint_ids]
    ).flatten()

    return model_joint_indxs


def get_model_actuator_ids(model: mj.MjModel, model_name: str) -> List[int]:
    """
    Retrieve the IDs of the actuators associated with a specified model in a MuJoCo simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    model_name : str
        The name of the model whose actuators are to be retrieved.

    Returns
    -------
    List[int]
        A list of actuator IDs associated with the specified model.

    Notes
    -----
    - The function filters actuator names by the given model_name and returns their corresponding IDs.
    """
    return [
        model.actuator(an).id for an in get_actuator_names(model) if model_name in an
    ]


def get_model_body_names(model: mj.MjModel, model_name: str) -> List[str]:
    """
    Retrieve the names of the bodies belonging to a specified model in a MuJoCo simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    model_name : str
        The name of the model whose bodies are to be retrieved.

    Returns
    -------
    List[str]
        A list of body names associated with the specified model.

    Notes
    -----
    - The function retrieves the body IDs associated with the specified model and maps them to their corresponding names.
    """
    return [body_id2name(model, bid) for bid in get_model_body_ids(model, model_name)]


def get_model_geom_names(model: mj.MjModel, model_name: str) -> List[str]:
    """
    Retrieve the names of the geometries associated with a specified model in a MuJoCo simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    model_name : str
        The name of the model whose geometries are to be retrieved.

    Returns
    -------
    List[str]
        A list of geometry names associated with the specified model.

    Notes
    -----
    - The function retrieves the geometry IDs associated with the specified model and maps them to their corresponding names.
    """
    return [geom_id2name(model, gid) for gid in get_model_geom_ids(model, model_name)]


def get_model_joint_names(model: mj.MjModel, model_name: str) -> List[str]:
    """
    Retrieve the names of the joints associated with a specified model in a MuJoCo simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    model_name : str
        The name of the model whose joints are to be retrieved.

    Returns
    -------
    List[str]
        A list of joint names associated with the specified model.

    Notes
    -----
    - The function retrieves the joint IDs associated with the specified model and maps them to their corresponding names.
    """
    return [joint_id2name(model, jid) for jid in get_model_joint_ids(model, model_name)]


def get_model_actuator_names(model: mj.MjModel, model_name: str) -> List[str]:
    """
    Retrieve the names of the actuators associated with a specified model in a MuJoCo simulation.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    model_name : str
        The name of the model whose actuators are to be retrieved.

    Returns
    -------
    List[str]
        A list of actuator names associated with the specified model.

    Notes
    -----
    - The function filters actuator names by the given model_name and returns their corresponding names.
    """
    return [
        model.actuator(an).name for an in get_actuator_names(model) if model_name in an
    ]


def get_actuator_forcerange(model: mj.MjModel, actuator_name: str) -> np.ndarray:
    """
    Retrieve the force range of the specified actuator in a MuJoCo simulation model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    actuator_name : str
        The name of the actuator whose force range is to be retrieved.

    Returns
    -------
    np.ndarray
        The force range of the specified actuator, typically a list containing the minimum and maximum force values.

    Notes
    -----
    - If the actuator_name is provided as a string, it will be converted to the corresponding actuator ID using `actuator_name2id`.
    - The function then returns the `forcerange` attribute for the specified actuator.
    """
    actuator_name = (
        actuator_name
        if isinstance(actuator_name, int)
        else actuator_name2id(model, actuator_name)
    )
    return model.actuator(actuator_name).forcerange


def is_mocap(model: mj.MjModel, body_name: Union[int, str]) -> bool:
    """
    Check if a given body in the MuJoCo model is a mocap (motion capture) body.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation structure.
    body_name : Union[int, str]
        The name or ID of the body to check. If a string is provided, it will be converted to the corresponding body ID.

    Returns
    -------
    bool
        True if the specified body is a mocap body, False otherwise.

    Notes
    -----
    - Mocap bodies in MuJoCo are bodies whose `body_mocapid` is greater than or equal to 0.
    """
    body_name = (
        body_name if isinstance(body_name, int) else body_name2id(model, body_name)
    )

    return model.body_mocapid[body_name] >= 0


def set_mocap_pose(
    data: mj.MjData, model: mj.MjModel, mocap_name: Union[int, str], T: sm.SE3
) -> None:
    """
    Set the pose of a motion capture (mocap) body in the MuJoCo simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current simulation state.
    model : mj.MjModel
        The MuJoCo model object defining the simulation structure.
    mocap_name : Union[int, str]
        The name (str) or ID (int) of the mocap body whose pose is to be set. If a string is provided, it is assumed to be the body name and will be converted to the corresponding body ID.
    T : sm.SE3
        The desired pose of the mocap body, represented as an SE3 object, where T.t is the translation (position) and T.R is the rotation (orientation).

    Notes
    -----
    - The function sets the `mocap_pos` (position) and `mocap_quat` (orientation) for the specified mocap body.
    - If the provided body is not a mocap body, a warning will be issued.
    """
    # convert name to id
    body_id = (
        mocap_name if isinstance(mocap_name, int) else body_name2id(model, mocap_name)
    )

    if is_mocap(model, mocap_name):
        mocap_id = model.body_mocapid[body_id]
        data.mocap_pos[mocap_id] = T.t
        data.mocap_quat[mocap_id] = smb.r2q(T.R)
    else:
        warnings.warn(
            f"The body with name '{mocap_name}' is not a mocap body.",
            UserWarning,
        )


def get_mocap_pose(
    data: mj.MjData, model: mj.MjModel, mocap_name: Union[int, str]
) -> sm.SE3:
    """
    Get the pose of a mocap (motion capture) object in the MuJoCo simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current simulation state.
    model : mj.MjModel
        The MuJoCo model object defining the simulation structure.
    mocap_name : Union[int, str]
        The name or ID of the mocap object whose pose is to be retrieved. If a string is provided, it will be converted to the corresponding body ID.

    Returns
    -------
    sm.SE3
        The pose (position and orientation) of the mocap object, represented as an SE3 object.

    Raises
    ------
    ValueError
        If the specified body is not a mocap object.

    Notes
    -----
    - The function checks if the given body is a mocap object using the `is_mocap` function.
    - If the body is a mocap object, the function retrieves its pose using `get_body_pose`.
    """
    # convert name to id
    body_id = (
        mocap_name if isinstance(mocap_name, int) else body_name2id(model, mocap_name)
    )
    if is_mocap(model, mocap_name):
        mocap_id = model.body_mocapid[body_id]
        return make_tf(pos=data.mocap_pos[mocap_id], ori=data.mocap_quat[mocap_id])
    else:
        raise ValueError(f"The body '{mocap_name}' is not a mocap object.")


def get_site_pose(
    data: mj.MjData, model: mj.MjModel, site_name: Union[int, str]
) -> sm.SE3:
    """
    Get the pose of a site in the MuJoCo simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current simulation state.
    model : mj.MjModel
        The MuJoCo model object defining the simulation structure.
    site_name : Union[int, str]
        The name or ID of the site whose pose is to be retrieved. If a string is provided, it will be converted to the corresponding site ID.

    Returns
    -------
    sm.SE3
        The pose (position and orientation) of the specified site, represented as an SE3 object.

    Raises
    ------
    ValueError
        If the specified site does not exist in the model.

    Notes
    -----
    - The function converts the site name to its corresponding ID if a string is provided.
    - The pose includes both the position and orientation of the site in the form of an SE3 object.
    """
    try:
        site_id = (
            site_name if isinstance(site_name, int) else site_name2id(model, site_name)
        )
        return make_tf(pos=data.site(site_id).xpos, ori=data.site(site_id).xmat)
    except Exception:
        raise ValueError(f"The site '{site_name}' does not exist in the model.")


def get_contact_states(
    data: mj.MjData, model: mj.MjModel, geom_name1: str = None, geom_name2: str = None
) -> List[ContactState]:
    """
    Retrieves the contact states for specified geometries in the MuJoCo simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object defining the simulation structure.
    geom_name1 : str, optional
        The name of the first geometry to check for contact. If provided, the function will filter the
        contact states to include only those involving this geometry. Default is None.
    geom_name2 : str, optional
        The name of the second geometry to check for contact. If provided, the function will filter the
        contact states to include only those involving this geometry. Default is None.

    Returns
    -------
    List[ContactState]
        A list of ContactState objects representing the states of the contacts. If specific geometry names
        are provided, the list will be filtered to include only the contacts involving the specified geometries.

    Notes
    -----
    - The function simulates forward dynamics to ensure that the contact states are up-to-date.
    - If both `geom_name1` and `geom_name2` are provided, the function returns only the contacts involving both geometries.
    - If only one geometry name is provided, the function returns all contacts involving that geometry.
    - If no geometry names are provided, the function returns all contact states in the simulation.

    Raises
    ------
    ValueError
        If the provided geometry names do not exist in the model.
    """

    # Get all contact states
    contact_states: List[ContactState] = []
    for i in range(data.ncon):
        contact = data.contact[i]
        cs = ContactState.from_mjcontact(contact, data, model, i)
        contact_states.append(cs)

    # Filter contacts based on the provided geometry names
    if geom_name1 is not None:
        geom_id1 = geom_name2id(model, geom_name1)
        contact_states = [
            cs for cs in contact_states if cs.geom1 == geom_id1 or cs.geom2 == geom_id1
        ]

    if geom_name2 is not None:
        geom_id2 = geom_name2id(model, geom_name2)
        contact_states = [
            cs
            for cs in contact_states
            if (cs.geom1 == geom_id2 or cs.geom2 == geom_id2)
            and (geom_name1 is None or cs.geom1 == geom_id1 or cs.geom2 == geom_id1)
        ]

    return contact_states


def apply_wrench(
    data: mj.MjData,
    model: mj.MjModel,
    body_name: str,
    wrench: Union[np.ndarray, List, Tuple],
) -> None:
    """
    Applies a wrench (force and torque) to a specific body in the MuJoCo simulation.

    The wrench is a 6-dimensional vector that includes both force (fx, fy, fz) and torque (tx, ty, tz).
    This function allows you to apply the wrench to a body in the simulation using its name.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the current simulation state.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    body_name : str
        The name of the body to which the wrench will be applied.
    wrench : Union[np.ndarray, List, Tuple]
        The wrench to be applied, provided as a 6-dimensional vector in any of the following formats:
        - numpy array
        - list
        - tuple

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the wrench is not a 6-dimensional vector or if the body name does not exist in the model.
    TypeError
        If the wrench is not a type that can be converted to a numpy array.
    """

    # Ensure the wrench is a numpy array
    if isinstance(wrench, (list, tuple)):
        try:
            wrench = np.array(wrench)
        except Exception as e:
            raise TypeError(f"Unable to convert wrench to numpy array: {e}")

    # Validate the wrench dimensions
    if wrench.shape != (6,):
        raise ValueError(
            "The wrench must be a 6-dimensional vector [fx, fy, fz, tx, ty, tz]."
        )

    # Get the body ID
    try:
        target_id = body_name2id(model, body_name)
    except Exception:
        raise ValueError(f"The body name '{body_name}' does not exist in the model.")

    # Apply the wrench to the specified body
    data.xfrc_applied[target_id, :] = wrench


def get_base_body_name(
    data: mj.MjData, model: mj.MjModel, robot_name: str
) -> Tuple[bool, str]:
    """
    Retrieve the name of the base body for a given robot in the MuJoCo model.

    This function searches through the body names in the MuJoCo model to find a body name
    that contains both "base" and the specified robot name. If such a body name is found,
    it is returned along with a boolean indicating success. If no such body name is found,
    an error is raised, and a tuple containing False and an empty string is returned.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object.
    model : mj.MjModel
        The MuJoCo model object.
    robot_name : str
        The name of the robot to search for.

    Returns
    -------
    Tuple[bool, str]:
        A tuple where the first element is a boolean indicating whether a base body name
        containing both "base" and the specified robot name was found, and the second
        element is the base body name if found, otherwise an empty string.

    Raises
    ------
    ValueError
        If no body name containing both "base" and the specified robot name is found.

    Notes
    -----
    Ensure that your model XML files include a body with a name containing both
    "base" and the specified robot name.
    """
    bns = get_body_names(model)

    # Check if "base" and robot_name are in each body name
    exists_conditions = [("base" in bn) and (robot_name in bn) for bn in bns]

    # Check if any of the conditions are True
    does_exist = np.any(exists_conditions)

    # Find the index where the condition is True
    indx = np.where(exists_conditions)[0]

    if not does_exist:
        raise ValueError(
            f'Body with name containing robot name "{robot_name}" and "base" could not be found. '
            "Make sure to add this to your XML files."
        )

    # If you need to return the base body name
    return (does_exist, bns[indx[0]])


def get_geoms_in_contact(data: mj.MjData, model: mj.MjModel) -> List[Tuple[str, str]]:
    """
    Retrieve a list of geometry pairs currently in contact in the MuJoCo simulation.

    This function iterates through all contacts in the MuJoCo simulation data and
    collects the names of geometries that are in contact. It returns a list of tuples,
    where each tuple contains the names of two geometries that are in contact.

    Args
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.

    Returns
    ----------
    List[Tuple[str, str]]:
        A list of tuples, where each tuple contains the names of two geometries that are in contact.

    Notes
    ----------
    Ensure that the MuJoCo model and data are correctly initialized before calling this function.
    The function assumes that all contact pairs are relevant and does not filter them.
    """
    contact_states = []

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = geom_id2name(model, contact.geom1)
        geom2_name = geom_id2name(model, contact.geom2)
        contact_states.append((geom1_name, geom2_name))

    return contact_states


def get_bodies_in_contact(data: mj.MjData, model: mj.MjModel) -> List[Tuple[str, str]]:
    geom_ids_in_contact = []

    for i in range(data.ncon):
        contact = data.contact[i]
        geom_ids_in_contact.append((contact.geom1, contact.geom2))

    body_names = [
        (
            body_id2name(model, model.geom_bodyid[gid[0]]),
            body_id2name(model, model.geom_bodyid[gid[1]]),
        )
        for gid in geom_ids_in_contact
    ]

    return body_names


def get_joint_pos(data: mj.MjData, model: mj.MjModel, joint_name: str) -> np.ndarray:
    """
    Retrieve the position of a specified joint in the MuJoCo simulation.

    This function returns the anchor position of the specified joint.

    Args:
    ----------
    data : mj.MjData
        The MuJoCo data object containing simulation data.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : str
        The name of the joint whose position is to be retrieved.

    Returns:
    ----------
    np.ndarray:
        The anchor position of the specified joint as a NumPy array.

    Notes:
    ----------
    Ensure that the specified joint name exists in the MuJoCo model.
    Raises a ValueError if the joint name is not found.
    """
    try:
        joint_id = joint_name2id(model, joint_name)
        return data.joint(joint_id).xanchor
    except Exception as e:
        raise ValueError(f"Joint '{joint_name}' not found in the model.") from e


def attach(
    data: mj.MjData,
    model: mj.MjModel,
    equality_name: str,
    gripper_freejoint_name: str,
    T_w_ee: sm.SE3,
    eq_data: np.ndarray = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ),
    eq_solimp: np.ndarray = np.array([[0.99, 0.99, 0.001, 0.5, 1]]),
    eq_solref: np.ndarray = np.array([0.0001, 1]),
) -> None:
    """
    Attaches a gripper to a robot's end-effector by setting up an equality constraint in the MuJoCo model.

    This function finds the specified equality constraint, disables it, moves the gripper's free joint
    to the robot's tool center point (TCP), sets the equality data, and then activates the constraint.

    Args:
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    equality_name : str
        The name of the equality constraint to attach.
    gripper_freejoint_name : str
        The name of the free joint associated with the gripper.
    T_w_ee : sm.SE3
        The pose of the robot's end-effector in the world frame, represented as an SE3 object.
    eq_data : np.ndarray, optional
        The data for the equality constraint (position, orientation, etc.). Default is an array with a specific configuration.
    eq_solimp : np.ndarray, optional
        The solimp parameters for the equality constraint. Default is an array with specific values.
    eq_solref : np.ndarray, optional
        The solref parameters for the equality constraint. Default is an array with specific values.

    Returns:
    ----------
    None

    Notes:
    ----------
    - The equality constraint specified by `equality_name` must exist in the model.
    - The gripper free joint should be properly defined in the model for the attachment to work correctly.
    - Ensure that the `T_w_ee` pose transformation is accurate and reflects the correct pose of the end-effector.
    - If the `equality_name` does not exist in the model, this function will raise a `ValueError`.
    """
    eq_id = eq_name2id(model, equality_name)

    if eq_id is None:
        raise ValueError(
            f"Equality constraint with name '{equality_name}' not found in the model."
        )

    # Ensure equality is disabled
    data.eq_active[eq_id] = 0

    # Move gripper free joint to robot TCP
    set_freejoint_pose(data, model, gripper_freejoint_name, T_w_ee)

    # Set equality data (pos, ori, etc.)
    model.equality(equality_name).data = eq_data

    # Set solref and solimp for equality constraint
    model.equality(equality_name).solimp = eq_solimp
    model.equality(equality_name).solref = eq_solref

    # Activate equality
    data.eq_active[eq_id] = 1


def detach(data: mj.MjData, model: mj.MjModel, equality_name: str) -> None:
    """
    Detaches a gripper from a robot by disabling the specified equality constraint in the MuJoCo model.

    This function finds the specified equality constraint and disables it, effectively detaching the gripper.

    Args:
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    equality_name : str
        The name of the equality constraint to detach.

    Returns:
    ----------
    None

    Notes:
    ----------
    - The equality constraint specified by `equality_name` must exist in the model.
    - If the `equality_name` does not exist, a `ValueError` will be raised.
    """
    eq_i = -1
    for i in range(model.neq):
        eq_name_i = eq_id2name(model, i)
        if eq_name_i == equality_name:
            eq_i = i
            break

    if eq_i == -1:
        raise ValueError(
            f"Equality constraint with name '{equality_name}' not found in the model."
        )

    # Disable equality
    data.eq_active[eq_i] = 0


def get_geom_distance(
    data: mj.MjData,
    model: mj.MjModel,
    geom1: Union[int, str],
    geom2: Union[int, str],
    distmax: float = 10.0,
) -> Tuple[float, np.ndarray]:
    """
    Calculate the smallest signed distance between two geometries (geoms) and the segment from one geom to the other.

    This function computes the distance between two specified geoms and provides the segment vector between them.
    The distance is computed within a specified maximum distance, `distmax`. If the distance is greater than
    `distmax`, the function will return `distmax`.

    Args:
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    geom1 : Union[int, str]
        The ID or name of the first geometry.
    geom2 : Union[int, str]
        The ID or name of the second geometry.
    distmax : float, optional
        The maximum distance for the distance calculation. Defaults to 10.0.

    Returns:
    ----------
    Tuple[float, np.ndarray]
        A tuple where the first element is the smallest signed distance between the two geoms, and the second
        element is the segment vector from `geom1` to `geom2`.

    Raises:
    ----------
    ValueError
        If `geom1` or `geom2` is not a valid geometry ID or name, or if `distmax` is not a positive number.

    Notes:
    ----------
    - Ensure that both `geom1` and `geom2` are valid geometry IDs or names within the MuJoCo model.
    - The distance calculation is sensitive to the model and simulation state. Ensure that the simulation is updated
      if necessary before calling this function.
    """

    if not isinstance(distmax, (int, float)) or distmax <= 0:
        raise ValueError(
            f"Invalid `distmax` value: {distmax}. It must be a positive number."
        )

    # Convert geom names to IDs if necessary
    geom1 = geom_name2id(model, geom1) if isinstance(geom1, str) else geom1
    geom2 = geom_name2id(model, geom2) if isinstance(geom2, str) else geom2

    # Initialize the segment vector
    from_to = np.zeros(6)

    # Calculate the distance and segment
    distance = mj.mj_geomDistance(model, data, geom1, geom2, distmax, from_to)

    return distance, from_to


def get_body_geoms(data: mj.MjData, model: mj.MjModel, body_name: Union[int, str]):
    """
    Retrieves the geometry IDs associated with a specific body in the MuJoCo simulation.

    This function finds all the geometries (geoms) that belong to the specified body,
    identified either by its name or ID.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object, which contains the state of the simulation.
    model : mj.MjModel
        The MuJoCo model object, which contains the simulation's structure.
    body_name : Union[int, str]
        The name or ID of the body for which the geometries are to be retrieved.

    Returns
    ----------
    list of int
        A list of geometry IDs associated with the specified body.
    """
    body_name = (
        body_name if isinstance(body_name, int) else body_name2id(model, body_name)
    )
    gids = get_geom_ids(model)
    body_gids = [gid for gid in gids if body_name == get_geom_body(model, gid)]
    return body_gids


def get_geom_body(model: mj.MjModel, geom_name: Union[int, str]):
    """
    Retrieves the body ID associated with a specific geometry in the MuJoCo simulation.

    This function identifies which body a geometry (geom) belongs to, based on the
    geometry's name or ID.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object, which contains the state of the simulation.
    model : mj.MjModel
        The MuJoCo model object, which contains the simulation's structure.
    geom_name : Union[int, str]
        The name or ID of the geometry for which the body ID is to be retrieved.

    Returns
    ----------
    int
        The body ID that the specified geometry is associated with.
    """
    geom_name = (
        geom_name if isinstance(geom_name, int) else geom_name2id(model, geom_name)
    )
    return model.geom(geom_name).bodyid[0]
