import math
import os
import pyrallis
import random
from dataclasses import dataclass, field
import multiprocessing as mp
import time
import pickle
from multiprocessing import shared_memory
from threading import Thread
from itertools import count
from colorama import Fore, Style
import cv2 as cv
import mujoco
import mujoco.viewer
import numpy as np
from common_utils import Stopwatch
from scipy.spatial.transform import Slerp
from envs.utils.arm_ik_solver import IKSolver
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from scipy.spatial.transform import Rotation as R
from constants import POLICY_CONTROL_PERIOD
from envs.utils.camera_utils import make_tf
import common_utils
import pdb
from types import SimpleNamespace


class BaseController:
    def __init__(self, qpos, qvel, ctrl, timestep):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl

        # OTG (online trajectory generation)
        num_dofs = 3
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]
        self.otg_res = None

    def reset(self):
        self.qpos[:] = np.array([
            0.0,  
            0.5,  
            0.0,   
        ])
        self.ctrl[:] = self.qpos

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            self.last_command_time = time.time()
            if 'base_pose' in command:
                self.otg_inp.target_position = command['base_pose']
                self.otg_res = Result.Working

        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.qpos
            self.otg_res = Result.Working

        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position


class ArmController:
    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep, reset_qpos, wbc=False):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper
        self.reset_qpos = reset_qpos

        num_dofs = 7
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
        self.otg_inp.max_acceleration = 4 * [math.radians(120)] + 3 * [math.radians(200)]
        self.otg_res = None

        self.wbc = wbc
        if not self.wbc:
            self.ik_solver = IKSolver(ee_offset=0.12)

    def reset(self):
        self.qpos[:] = np.array(self.reset_qpos)
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = 0.0

        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            self.last_command_time = time.time()

            if 'arm_qpos' in command:
                self.otg_inp.target_position = command['arm_qpos']
                self.otg_res = Result.Working

            elif 'arm_pos' in command:
                qpos = self.ik_solver.solve(command['arm_pos'], command['arm_quat'], self.qpos)
                qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi
                self.otg_inp.target_position = qpos
                self.otg_res = Result.Working

            if 'gripper_pos' in command:
                self.ctrl_gripper[:] = 255.0 * command['gripper_pos']

        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.otg_out.new_position
            self.otg_res = Result.Working

        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position


class ShmState:
    def __init__(self, existing_instance=None):
        # 移除了 force (3) 和 torque (3)，长度从原本的缩减
        arr = np.empty(3 + 3 + 4 + 1 + 1 + 1 + 1)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.base_pose = self.data[:3]
        self.arm_pos = self.data[3:6]
        self.arm_quat = self.data[6:10]
        self.gripper_pos = self.data[10:11]
        self.initialized = self.data[11:12]
        self.initialized[:] = 0.0
        self.reward = self.data[12:13]
        self.reward[:] = 0
        self.goal_cube = self.data[13:14] 
        self.goal_cube[:] = 1.0  

    def close(self):
        self.shm.close()


class ShmCameraParameters:
    def __init__(self, existing_instance=None):
        arr = np.empty(4*4 + 3*3)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.extrinsics = self.data[:4*4].reshape(4,4)
        self.intrinsics = self.data[4*4:].reshape(3,3)

    def close(self):
        self.shm.close()


class ShmImage:
    def __init__(self, camera_name=None, width=None, height=None, channels=3, existing_instance=None):
        if existing_instance is None:
            self.camera_name = camera_name
            self.channels = channels
            dtype = np.uint8 if channels == 3 else np.float32
            self.shm = shared_memory.SharedMemory(create=True, size=width * height * channels * np.dtype(dtype).itemsize)
            shape = (height, width, channels) if channels == 3 else (height, width)
            self.data = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        else:
            self.camera_name = existing_instance.camera_name
            self.channels = existing_instance.channels
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
            shape = existing_instance.data.shape
            self.data = np.ndarray(shape, dtype=existing_instance.data.dtype, buffer=self.shm.buf)
        self.data.fill(0)

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()


class Renderer:
    def __init__(self, model, data, shm_image, shm_depth, shm_cam_params):
        self.model = model
        self.data = data
        self.image = np.empty_like(shm_image.data)
        self.depth = np.empty((shm_image.data.shape[0], shm_image.data.shape[1]), dtype=np.float32)

        self.shm_image = ShmImage(existing_instance=shm_image)
        self.shm_depth = ShmImage(existing_instance=shm_depth)
        self.shm_cam_params = ShmCameraParameters(existing_instance=shm_cam_params)

        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA.value, shm_image.camera_name)
        self.camera_id = camera_id
        width, height = model.cam_resolution[camera_id]
        self.width = width
        self.height = height
        self.camera = mujoco.MjvCamera()
        self.camera.fixedcamid = camera_id
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera_name = shm_image.camera_name

        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.gl_context = mujoco.gl_context.GLContext(width, height)
        self.gl_context.make_current()
        self.mjr_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context)

        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, 10000)

    def render(self):
        original_znear = self.model.vis.map.znear
        self.model.vis.map.znear = 0.05 / self.model.stat.extent

        self.gl_context.make_current()
        mujoco.mjv_updateScene(
            self.model, self.data, self.scene_option, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
        )
        mujoco.mjr_render(self.rect, self.scene, self.mjr_context)

        mujoco.mjr_readPixels(self.image, self.depth, self.rect, self.mjr_context)
        self.shm_image.data[:] = np.flipud(self.image)

        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        self.depth = near / (1 - self.depth * (1 - near / far))
        self.shm_depth.data[:] = np.flipud(self.depth)

        self.model.vis.map.znear = original_znear

    def get_params(self):
        aspect_ratio = self.width / self.height
        fovy = np.radians(self.model.cam_fovy[self.camera_id])
        fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)
        fx, fy = self.width / (2 * np.tan(fovx / 2)), self.height / (2 * np.tan(fovy / 2))
        cx, cy = self.width / 2, self.height / 2

        intrinsics = np.array([
            [fx, 0, cx],  
            [0, fy, cy],  
            [0, 0, 1]  
        ])

        cam_pos = self.model.cam_pos[self.camera_id]  
        cam_quat = self.model.cam_quat[self.camera_id]  

        cam_rot = np.zeros((9,))
        mujoco.mju_quat2Mat(cam_rot, cam_quat)
        cam_rot = cam_rot.reshape((3, 3))

        body_id = self.model.cam_bodyid[self.camera_id]  
        body_pos = self.data.xpos[body_id]  
        body_quat = self.data.xquat[body_id]  

        body_rot = np.zeros((9,))
        mujoco.mju_quat2Mat(body_rot, body_quat)
        body_rot = body_rot.reshape((3, 3))

        global_cam_pos = body_pos + body_rot @ cam_pos
        global_cam_rot = body_rot @ cam_rot

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = global_cam_rot
        extrinsics[:3, 3] = global_cam_pos
        self.shm_cam_params.data[:] = np.hstack((extrinsics.flatten(), intrinsics.flatten()))

    def close(self):
        self.gl_context.free()
        self.gl_context = None
        self.mjr_context.free()
        self.mjr_context = None


class CommonMujocoSim:
    def __init__(self, task, mjcf_path, command_queue, shm_state, show_viewer=True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.vis.map.znear = 0.05
        self.model.vis.map.zfar = 10.0
        self.data = mujoco.MjData(self.model)
        self.command_queue = command_queue
        self.show_viewer = show_viewer

        self.sensor = None
        self.task = task
        assert self.task in ["cube", "cube_size", "cube_distractor", "cube_specified", "open", "dishwasher", "door", "bare"]

        self.model.body_gravcomp[:] = 1.0
        body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        for object_name in ['interactive_obj']:
            if object_name in body_names:
                self.model.body_gravcomp[self.model.body(object_name).id] = 0.0

        self.base_dofs = base_dofs = self.model.body('base_link').jntnum.item()
        self.qpos_base = self.data.qpos[:base_dofs]
        qvel_base = self.data.qvel[:base_dofs]
        ctrl_base = self.data.ctrl[:base_dofs]

        self.base_controller = BaseController(self.qpos_base, qvel_base, ctrl_base, self.model.opt.timestep)
        self.shm_state = ShmState(existing_instance=shm_state)

        site_id = self.model.site('pinch_site').id
        self.site_xpos = self.data.site(site_id).xpos
        self.site_xmat = self.data.site(site_id).xmat
        self.site_quat = np.empty(4)
        self.base_height = self.model.body('gen3/base_link').pos[2]
        self.arm_forward = self.model.body('gen3/base_link').pos[0]
        self.base_rot_axis = np.array([0.0, 0.0, 1.0])
        self.base_quat_inv = np.empty(4)

        self.stain_geom_ids = []
        self.stain_geom_names = []
        self.stain_geom_rgba = np.empty((0, 4), dtype=np.float64)
        self.stain_active = np.zeros((0,), dtype=bool)
        self.pad_geom_ids = []
        self.stain_erase_dist = 0.05
        self._init_door_wipe_state()

    def _init_door_wipe_state(self):
        if self.task != "door":
            return

        stain_pairs = []
        for geom_id in range(self.model.ngeom):
            geom_name = self.model.geom(geom_id).name
            if geom_name is not None and geom_name.startswith("stain_"):
                stain_pairs.append((geom_name, geom_id))
        stain_pairs.sort(key=lambda x: x[0])

        if stain_pairs:
            self.stain_geom_names = [name for name, _ in stain_pairs]
            self.stain_geom_ids = [geom_id for _, geom_id in stain_pairs]
            self.stain_geom_rgba = self.model.geom_rgba[self.stain_geom_ids].copy()
            self.stain_active = np.ones((len(self.stain_geom_ids),), dtype=bool)

        preferred_pad_names = ("left_pad1", "left_pad2")
        fallback_pad_names = ("gelsight_col_left", "gelsight_vis_left")
        for pad_name in preferred_pad_names + fallback_pad_names:
            pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, pad_name)
            if pad_id != -1 and pad_id not in self.pad_geom_ids:
                self.pad_geom_ids.append(pad_id)

    def _reset_door_stains(self):
        if self.task != "door" or len(self.stain_geom_ids) == 0:
            return
        self.stain_active[:] = True
        for i, geom_id in enumerate(self.stain_geom_ids):
            self.model.geom_rgba[geom_id] = self.stain_geom_rgba[i]
            self.model.geom_pos[geom_id] = np.array([
                -0.051,                        
                np.random.uniform(-0.25, 0.25),
                np.random.uniform(0.7, 1.1),
            ])

    def _update_door_stains(self):
        if self.task != "door" or len(self.stain_geom_ids) == 0 or len(self.pad_geom_ids) == 0:
            return

        pad_positions = self.data.geom_xpos[self.pad_geom_ids]
        for i, geom_id in enumerate(self.stain_geom_ids):
            if not self.stain_active[i]:
                continue
            stain_pos = self.data.geom_xpos[geom_id]
            min_dist = np.min(np.linalg.norm(pad_positions - stain_pos, axis=1))
            if min_dist < self.stain_erase_dist:
                self.stain_active[i] = False
                self.model.geom_rgba[geom_id, 3] = 0.0

    def update_shm_state(self):
        self._update_door_stains()
        self.shm_state.base_pose[:] = self.qpos_base

        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height  
        site_xpos[:2] -= self.qpos_base[:2]  
        mujoco.mju_axisAngle2Quat(self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2])  
        mujoco.mju_rotVecQuat(self.shm_state.arm_pos, site_xpos, self.base_quat_inv)  
        self.shm_state.arm_pos[0] -= self.arm_forward

        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
        mujoco.mju_mulQuat(self.shm_state.arm_quat, self.base_quat_inv, self.site_quat)  

        self.shm_state.gripper_pos[:] = self.qpos_gripper / 0.8  

        # 力传感器采集代码已移除

        self.shm_state.initialized[:] = 1.0
        self.shm_state.reward[:] = self.is_success()

    def reset_task(self):
        if self.task == "cube":
            randomized_position = np.random.uniform(low=(0.5, -0.2, 0), high=(1.3, 0.2, 0), size=3)
            randomized_position[2] = 0.05  
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            self.data.xpos[interactive_body_id] += randomized_position
            self.data.qpos[self.model.joint("interactive_obj_freejoint").id : self.model.joint("interactive_obj_freejoint").id + 3] += randomized_position

        elif self.task == "cube_size":
            randomized_position = np.random.uniform(low=(0.5, -0.2, 0), high=(1.0, 0.2, 0), size=3)
            randomized_position[2] = 0.05  
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            self.data.xpos[interactive_body_id] += randomized_position
            self.data.qpos[self.model.joint("interactive_obj_freejoint").id : self.model.joint("interactive_obj_freejoint").id + 3] += randomized_position
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
            size = np.random.uniform(low=[0.02, 0.01, 0.01], high=[0.04, 0.0275, 0.0275], size=3)
            self.model.geom_size[geom_id] = size

        elif self.task == "cube_distractor":
            cube_low = np.array([0.5, -0.2, 0.1])
            cube_high = np.array([1.0, 0.2, 0.1])  
            distractor_low = np.array([0.6, -0.2, 0.1])
            distractor_high = np.array([0.9, 0.2, 0.1])  
            min_distance = 0.05
            
            cube_pos = np.random.uniform(low=cube_low, high=cube_high)
            
            distractor_positions = []
            for _ in range(3):
                for _ in range(500):
                    candidate = np.random.uniform(low=distractor_low, high=distractor_high)
                    if all(np.linalg.norm(candidate[:2] - pos[:2]) >= min_distance for pos in distractor_positions):
                        distractor_positions.append(candidate)
                        break
                else:
                    raise RuntimeError("Failed to sample distractors with required separation.")
            
            positions = [cube_pos] + distractor_positions
            bodies_and_joints = [
                ("interactive_obj", "interactive_obj_freejoint"),
                ("distractor1", "distractor1_freejoint"),
                ("distractor2", "distractor2_freejoint"),
                ("distractor3", "distractor3_freejoint"),
            ]
            
            for pos, (body_name, joint_name) in zip(positions, bodies_and_joints):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_adr : qpos_adr + 3] = pos
            mujoco.mj_forward(self.model, self.data)

        elif self.task == "cube_specified":
            low = np.array([0.5, -0.2, 0.1])  
            high = np.array([0.9, 0.2, 0.1])
            min_distance = 0.02  
            
            cube1 = np.random.uniform(low=low, high=high)
            for _ in range(100):
                cube2 = np.random.uniform(low=low, high=high)
                if np.linalg.norm(cube2[:2] - cube1[:2]) >= min_distance:
                    break
            else:
                raise RuntimeError("Failed to sample two cubes with required separation.")
            
            joint1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "interactive_obj_freejoint")
            joint2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "interactive_obj2_freejoint")
            qpos1_adr = self.model.jnt_qposadr[joint1_id]
            qpos2_adr = self.model.jnt_qposadr[joint2_id]
            
            self.data.qpos[qpos1_adr : qpos1_adr + 3] = cube1
            self.data.qpos[qpos2_adr : qpos2_adr + 3] = cube2
            mujoco.mj_forward(self.model, self.data)

        elif self.task == "open":
            randomized_position = np.random.uniform(low=(0.1, -0.5, 0), high=(0.8, 0.5, 0), size=3)
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            self.data.xpos[interactive_body_id] += randomized_position
            self.data.qpos[self.model.joint("interactive_obj_freejoint").id : self.model.joint("interactive_obj_freejoint").id + 3] += randomized_position

        elif self.task == "dishwasher":
            randomized_position = np.random.uniform(low=(0.1, -0.5, 0), high=(0.7, 0.5, 0), size=3)
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            self.data.xpos[interactive_body_id] += randomized_position
            self.data.qpos[self.model.joint("interactive_obj_freejoint").id : self.model.joint("interactive_obj_freejoint").id + 3] += randomized_position

        elif self.task == "door":
            door_cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_cube_freejoint")
            if door_cube_joint_id != -1:
                qpos_adr = self.model.jnt_qposadr[door_cube_joint_id]
                base_pos = np.array([0.9, 0.5, 1.1])
                self.data.qpos[qpos_adr : qpos_adr + 3] = np.array([
                    base_pos[0] + np.random.uniform(-0.01, 0.01),
                    base_pos[1] + np.random.uniform(-0.01, 0.01),
                    base_pos[2] + np.random.uniform(-0.005, 0.005),
                ])
            self._reset_door_stains()

    def is_success(self):
        if self.task == "bare":
            return 0
        if self.task in ["cube", "cube_size", "cube_distractor"]:
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "interactive_obj")
            cube_pos = self.data.xpos[interactive_body_id]
            z_thresh = 0.10
            reward = (cube_pos[2] > z_thresh)
        elif self.task in ["cube_specified"]:
            goal_is_green = (self.shm_state.goal_cube[0] == 1.0)
            id_name = "interactive_obj" if goal_is_green else "interactive_obj2"
            interactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, id_name)
            cube_pos = self.data.xpos[interactive_body_id]
            z_thresh = 0.10
            reward = (cube_pos[2] > z_thresh)
        elif self.task == "open":
            door_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rightdoorhinge")
            right_door_angle = self.data.sensordata[door_id]
            angle_thresh = 0.5
            reward = right_door_angle > angle_thresh
        elif self.task == "dishwasher":
            door_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "door")
            door_angle = self.data.sensordata[door_id]
            assert -np.pi / 2 < door_angle < 0.1
            angle_thresh = -np.pi / 8
            reward = door_angle < angle_thresh
        elif self.task == "door":
            if len(self.stain_geom_ids) > 0:
                reward = bool(np.all(~self.stain_active))
            else:
                door_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "door_hinge")
                door_angle = self.data.sensordata[door_id]
                angle_thresh = 0.7
                reward = door_angle > angle_thresh
        return reward

    def reset(self):
        pass

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def launch(self):
        if self.show_viewer:
            mujoco.viewer.launch(self.model, self.data, show_left_ui=False, show_right_ui=False)
        else:
            last_step_time = 0
            while True:
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                try:
                    mujoco.mj_step(self.model, self.data)
                except:
                    import pdb; pdb.set_trace()

@dataclass
class MujocoEnvConfig:
    wbc: int
    cameras: list[str]
    task: str
    data_folder: str
    is_sim: int = 1
    arm_reset_qpos: list[float] = field(default_factory=lambda: [
        -0.011519237, -0.313874840, -3.119614118, -1.488434342, 3.102002900, 0.531296044, -1.534899046
    ])
    min_bound: list[float] = field(default_factory=list)
    max_bound: list[float] = field(default_factory=list)

class CommonMujocoEnv:
    def __init__(self, cfg: MujocoEnvConfig, render_images=True, show_viewer=True, show_images=False):
        self.cfg = cfg
        self.ARM_BASE_OFFSET = [0.1199, 0, 0.3948]

        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.command_queue = mp.Queue(1)
        self.num_step = 0
        self.reward = 0

        self.task = self.cfg.task
        assert self.task in ["cube", "cube_size", "cube_distractor", "cube_specified", "open", "dishwasher", "door", "bare"]

        TASK_TO_MJCF_PATH = {
            'cube': "mj_assets/stanford_tidybot2/cube.xml",
            'cube_size': "mj_assets/stanford_tidybot2/cube_size.xml",
            'cube_distractor': "mj_assets/stanford_tidybot2/cube_distractor.xml",
            'cube_specified': "mj_assets/stanford_tidybot2/cube_specified.xml",
            'open': "mj_assets/stanford_tidybot2/open.xml",
            'dishwasher': "mj_assets/stanford_tidybot2/dishwasher.xml",
            'door': "mj_assets/stanford_tidybot2/door.xml",
            'bare': "mj_assets/stanford_tidybot2/bare_robot.xml"
        }

        self.mjcf_path = TASK_TO_MJCF_PATH[self.cfg.task]

        self.shm_state = ShmState()
        self.shm_cam_params = []
        self.shm_images_rgb = []
        self.shm_images_depth = []
        self.camera_names = []

        self.teleop_policy = None
        self.offscreen_warn_threshold_s = 0.1
        self.offscreen_warn_cooldown_s = 5.0
        self._last_offscreen_warn_time = 0.0
        self.cam_params_update_stride = 5

        if self.render_images:
            for camera_name in self.cfg.cameras:
                self.shm_cam_params.append(ShmCameraParameters())
                self.camera_names.append(camera_name)
                self.shm_images_rgb.append(ShmImage(camera_name, 640, 480, channels=3))
                self.shm_images_depth.append(ShmImage(camera_name, 640, 480, channels=1))

        if self.render_images and self.show_images:
            self.visualizer_process = mp.Process(target=self.visualizer_loop, daemon=True)
            self.visualizer_process.start()
        
    def _dump_or_check_env_cfg(self):
        cfg_path = os.path.join(self.cfg.data_folder, "env_cfg.yaml")
        if not os.path.exists(cfg_path):
            print(f"saving env cfg to {cfg_path}")
            pyrallis.dump(self.cfg, open(cfg_path, "w"))  
        else:
            assert common_utils.check_cfg(MujocoEnvConfig, cfg_path, self.cfg), \
                f"Error: {self.cfg.data_folder} contains a different config than the current one"

    def physics_loop(self):
        pass

    def render_loop(self, model, data):
        renderers = [Renderer(model, data, shm_image, shm_depth_image, shm_cam_params) for (shm_image, shm_depth_image, shm_cam_params) in zip(self.shm_images_rgb, self.shm_images_depth, self.shm_cam_params)]
        frame_idx = 0

        while True:
            start_time = time.time()
            camera_render_time = 0.0
            
            for renderer in renderers:
                cam_t0 = time.time()
                renderer.render()
                if frame_idx % self.cam_params_update_stride == 0:
                    renderer.get_params()
                camera_render_time += time.time() - cam_t0

            frame_idx += 1
            render_time = time.time() - start_time
            now = time.time()
            
            if (
                render_time > self.offscreen_warn_threshold_s
                and now - self._last_offscreen_warn_time > self.offscreen_warn_cooldown_s
            ):
                self._last_offscreen_warn_time = now
                fps = 1.0 / max(render_time, 1e-6)
                print(
                    f"Warning: Offscreen rendering took {1000 * render_time:.1f} ms "
                    f"({fps:.1f} FPS, cameras={len(renderers)})."
                )

    def visualizer_loop(self):
        shm_images = [ShmImage(existing_instance=shm_image) for shm_image in self.shm_images_rgb]
        last_imshow_time = time.time()
        while True:
            while time.time() - last_imshow_time < 0.1:  
                time.sleep(0.01)
            last_imshow_time = time.time()
            for i, shm_image in enumerate(shm_images[1:]):
                image = shm_image.data
                resized_image = cv.resize(image, (480 * 2, 360 * 2))
                cv.imshow(shm_image.camera_name, cv.cvtColor(resized_image, cv.COLOR_RGB2BGR))
                if i < 3:
                    cv.moveWindow(shm_image.camera_name, 480 * 2 * i, -100)
                else:
                    cv.moveWindow(shm_image.camera_name, 480 * 2 * (i - 3), 480 * 2)
            cv.waitKey(1)

    def reset(self):
        self.num_step = 0
        self.shm_state.initialized[:] = 0.0
        self.command_queue.put('reset')
        while self.shm_state.initialized == 0.0:
            time.sleep(0.01)

        if self.render_images:
            while any(np.all(shm_image.data == 0) for shm_image in self.shm_images_rgb):
                time.sleep(0.01)

        if self.task == 'cube_specified':
            self.goal_cube = random.choice(["green", "red"])
            self.shm_state.goal_cube[:] = 1.0 if self.goal_cube == "green" else 0.0
            return self.goal_cube
        elif 'cube' in self.task:
            self.goal_cube = "green"
            return self.goal_cube
        return None

    def get_obs(self):
        base_pose = self.shm_state.base_pose.copy()
        arm_pos = self.shm_state.arm_pos.copy()
        arm_quat = self.shm_state.arm_quat[[1, 2, 3, 0]]  
        if arm_quat[3] < 0.0:  
            np.negative(arm_quat, out=arm_quat)

        arm_pos_global = self.local_to_global_arm_pos(arm_pos, base_pose)
        gripper_pos = self.shm_state.gripper_pos.copy()

        obs = {
            'base_pose': base_pose,
            'arm_pos': arm_pos_global,
            'arm_quat': arm_quat,
            'gripper_pos': gripper_pos,
            'reward': self.shm_state.reward.copy(),
            'proprio': np.hstack((arm_pos, arm_quat, gripper_pos, base_pose))
        }

        if self.render_images:
            camera_axis_correction = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            for (name, shm_rgb, shm_depth, shm_cam_params) in zip(self.camera_names, self.shm_images_rgb, self.shm_images_depth, self.shm_cam_params):
                obs[f'{name}_image'] = shm_rgb.data.copy()
                obs[f'{name}_depth'] = shm_depth.data.copy()
                cam_params = shm_cam_params.data.copy()
                obs["%s_T" % name] = cam_params[:4*4].reshape((4,4)) @ camera_axis_correction
                obs["%s_K" % name] = cam_params[4*4:].reshape((3,3))

        return obs

    def seed(self, seed):
        self.command_queue.put(("set_seed", seed))
        time.sleep(0.1)

    def step(self, action):
        self.command_queue.put(action)
        self.num_step += 1

    def close(self):
        self.shm_state.close()
        self.shm_state.shm.unlink()

        if self.render_images:
            for shm_rgb, shm_depth, shm_cam_param in zip(self.shm_images_rgb, self.shm_images_depth, self.shm_cam_params):
                shm_rgb.close()
                shm_rgb.shm.unlink()
                shm_depth.close()
                shm_depth.shm.unlink()
                shm_cam_param.close()
                shm_cam_param.shm.unlink()

        if hasattr(self, 'visualizer_process') and self.visualizer_process.is_alive():
            self.visualizer_process.terminate()
            self.visualizer_process.join()

    def local_to_global_arm_pos(self, arm_pos_local, base_pose):
        T_base_world = np.eye(4)
        T_base_world[:3, :3] = R.from_euler('z', base_pose[2]).as_matrix()
        T_base_world[:3, 3] = np.array([base_pose[0], base_pose[1], 0]) + self.ARM_BASE_OFFSET
        arm_pos_global = T_base_world@np.array([arm_pos_local[0], arm_pos_local[1], arm_pos_local[2], 1.0])
        arm_pos_global = arm_pos_global[:3]
        return arm_pos_global

    def global_to_local_arm_pos(self, arm_pos_global, base_pose):
        T_base_world = np.eye(4)
        T_base_world[:3, :3] = R.from_euler('z', base_pose[2]).as_matrix()
        T_base_world[:3, 3] = np.array([base_pose[0], base_pose[1], 0]) + self.ARM_BASE_OFFSET
        arm_pos_local = np.linalg.inv(T_base_world)@np.array([arm_pos_global[0], arm_pos_global[1], arm_pos_global[2], 1.0])
        arm_pos_local = arm_pos_local[:3]
        return arm_pos_local

    def move_to_base_waypoint(self, target_base_pose, threshold_pos=0.01, threshold_theta=0.01):
        obs = self.get_obs()
        curr_base_pose = obs["base_pose"]

        while True:
            obs = self.get_obs()
            curr_base_pose = obs["base_pose"]

            pos_error_norm = np.linalg.norm(curr_base_pose[:2] - target_base_pose[:2])
            theta_error = target_base_pose[2] - curr_base_pose[2]

            if pos_error_norm < threshold_pos and abs(theta_error) < threshold_theta:
                return True, pos_error_norm  

            self.step({"base_pose": target_base_pose})
            time.sleep(POLICY_CONTROL_PERIOD)  

        return False, pos_error_norm

    def move_to_arm_waypoint(self, target_arm_pos, target_arm_quat, target_gripper_pos, step_size=0.1, threshold_pos=0.01, threshold_quat=0.01):
        if target_arm_quat[3] < 0:
            np.negative(target_arm_quat, out=target_arm_quat)

        reached = False
        pos_error_norm = np.inf
        MAX_STEP = 50
        step = 0

        while not reached:
            obs = self.get_obs()
            curr_arm_pos, curr_arm_quat = obs["arm_pos"], obs["arm_quat"]

            pos_error = target_arm_pos - curr_arm_pos
            pos_error_norm = np.linalg.norm(pos_error)

            quat_error = 1 - abs(np.dot(curr_arm_quat, target_arm_quat))

            if pos_error_norm < threshold_pos and quat_error < threshold_quat:
                reached = True
                break
            elif step > MAX_STEP:
                break

            step_vec = step_size * pos_error / (pos_error_norm + 1e-6)  
            next_pos = curr_arm_pos + step_vec if pos_error_norm > step_size else target_arm_pos

            if not self.cfg.wbc:
                next_pos = self.global_to_local_arm_pos(next_pos, obs['base_pose'])

            key_times = [0, 1]  
            key_rots = R.from_quat([curr_arm_quat, target_arm_quat])  
            slerp = Slerp(key_times, key_rots)  
            interp_ratio = min(step_size / (pos_error_norm + 1e-6), 1.0)  
            next_quat = slerp([interp_ratio]).as_quat()[0]  

            self.step({
                "arm_pos": next_pos,
                "arm_quat": next_quat,
                "gripper_pos": 1 if obs['gripper_pos'] > 0.3 else obs['gripper_pos'],
            })

            time.sleep(POLICY_CONTROL_PERIOD)  
            step += 1

        for _ in range(10):  
            obs = self.get_obs()
            self.step({"gripper_pos": target_gripper_pos})
            time.sleep(POLICY_CONTROL_PERIOD)

        return reached, pos_error_norm