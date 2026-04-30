import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.utils.util import quat2theta, relative_position, relative_quaternion


_ROOT = Path(__file__).resolve().parents[2]
_RMM_PATH = _ROOT / "mujoco_command"
if str(_RMM_PATH) not in sys.path:
    sys.path.append(str(_RMM_PATH))

from rmm_api import RobotAPI  # noqa: E402


class RealRobotBridgeEnv:
    """
    Bridge the current Tracking policy interface to the lab RobotAPI interface.

    This adapter keeps the current action convention:
    - base: 2D planar velocity
    - arm:  7D joint delta
    - hand: 1D gripper command

    and translates it into RobotAPI.step(...):
    - base_pose
    - arm_pos
    - arm_quat
    - gripper_pos

    Notes:
    - Object observations are intentionally left as placeholders for now.
    - A local shadow MuJoCo model is used only for kinematic translation.
    """

    def __init__(self, robot_api=None, robot_cfg_path="envs/cfgs/door_wbc.yaml", control_dt=0.02):
        self.task = "Tracking"
        self.robot = robot_api if robot_api is not None else RobotAPI(cfg_path=robot_cfg_path)
        self.control_dt = float(control_dt)

        model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_LEAP_OBJECT_PATH)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.arm_joint_names = [f"joint_{i}" for i in range(1, 8)]
        self.arm_qpos_indices = np.array(
            [self.model.joint(name).qposadr[0] for name in self.arm_joint_names], dtype=int
        )
        self.base_qpos_indices = np.array(
            [self.model.joint(name).qposadr[0] for name in ["joint_x", "joint_y", "joint_th"]], dtype=int
        )
        self.ee_site_id = self.model.site(DcmmCfg.ee_site_name).id
        self.ee_body_id = self.model.body(DcmmCfg.ee_body_name).id

        self.arm_base_offset = np.array([0.1199, 0.0, 0.3948], dtype=np.float32)
        self.base_vel_lpf_alpha = 1.0
        self.target_base_vel = np.zeros(2, dtype=np.float32)

        self.object_pos3d = np.zeros(3, dtype=np.float32)
        self.object_vel3d = np.zeros(3, dtype=np.float32)

        self.reset()

    def call(self, name):
        return [getattr(self, name)]

    def close(self):
        close_fn = getattr(self.robot, "close", None)
        if callable(close_fn):
            close_fn()

    def set_object_state(self, pos3d=None, v_lin_3d=None):
        if pos3d is not None:
            self.object_pos3d = np.asarray(pos3d, dtype=np.float32).copy()
        if v_lin_3d is not None:
            self.object_vel3d = np.asarray(v_lin_3d, dtype=np.float32).copy()

    def reset(self):
        self.shadow_base_pose = np.array(DcmmCfg.base_init_pose, dtype=np.float32)
        self.shadow_arm_qpos = np.array(DcmmCfg.arm_joints, dtype=np.float32)
        self.shadow_gripper = float(DcmmCfg.hand_joints[0])
        self.target_base_vel[:] = 0.0

        self._sync_shadow_model()
        self.prev_time = time.time()
        self.prev_base_pose = self.shadow_base_pose.copy()
        self.prev_ee_rel_pos = self._shadow_relative_ee_pos().copy()

        obs = self.get_obs()
        return obs, {}

    def step(self, action_dict):
        action_dict = {
            k: np.asarray(v, dtype=np.float32).squeeze()
            for k, v in action_dict.items()
        }

        base_action = np.asarray(action_dict.get("base", np.zeros(2)), dtype=np.float32).reshape(-1)
        arm_action = np.asarray(action_dict.get("arm", np.zeros(7)), dtype=np.float32).reshape(-1)
        hand_action = np.asarray(action_dict.get("hand", np.zeros(1)), dtype=np.float32).reshape(-1)

        self.target_base_vel *= 1.0 - self.base_vel_lpf_alpha
        self.target_base_vel += self.base_vel_lpf_alpha * base_action[:2]
        self.shadow_base_pose[:2] += self.target_base_vel[:2] * self.control_dt

        self.shadow_arm_qpos = self._clip_arm_qpos(self.shadow_arm_qpos + arm_action[:7])
        if hand_action.size > 0:
            self.shadow_gripper = float(np.clip(hand_action[0], 0.0, 1.0))

        self._sync_shadow_model()
        target_arm_pos = self._shadow_ee_world_pos()
        target_arm_quat = self._shadow_ee_world_quat_xyzw()

        self.robot.step(
            arm_pos=target_arm_pos,
            arm_quat=target_arm_quat,
            gripper_pos=self.shadow_gripper,
            base_pose=self.shadow_base_pose.copy(),
        )

        obs = self.get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def get_obs(self):
        state = self.robot.get_state()
        now = time.time()
        dt = max(now - self.prev_time, 1e-6)

        base_pose = np.asarray(
            state.get("base_pose", self.shadow_base_pose), dtype=np.float32
        ).copy()
        arm_pos_world = np.asarray(state["arm_pos"], dtype=np.float32).copy()
        arm_quat_xyzw = np.asarray(state["arm_quat"], dtype=np.float32).copy()
        gripper_pos = float(state.get("gripper_pos", self.shadow_gripper))
        joint_pos = np.asarray(state.get("joint_pos", self.shadow_arm_qpos), dtype=np.float32).copy()

        base_vel = ((base_pose[:2] - self.prev_base_pose[:2]) / dt).astype(np.float32)
        arm_rel_pos = self._relative_ee_pos_from_world(base_pose, arm_pos_world)
        arm_rel_quat = self._relative_ee_quat_from_world(base_pose, arm_quat_xyzw)
        arm_rel_vel = ((arm_rel_pos - self.prev_ee_rel_pos) / dt).astype(np.float32)

        obs = {
            "base": {
                "v_lin_2d": base_vel,
            },
            "arm": {
                "ee_pos3d": arm_rel_pos,
                "ee_quat": arm_rel_quat,
                "ee_v_lin_3d": arm_rel_vel,
                "joint_pos": joint_pos,
            },
            "hand": np.array([gripper_pos], dtype=np.float32),
            "object": {
                "pos3d": self.object_pos3d.copy(),
                "v_lin_3d": self.object_vel3d.copy(),
            },
        }

        self.prev_time = now
        self.prev_base_pose = base_pose
        self.prev_ee_rel_pos = arm_rel_pos
        self.shadow_base_pose = base_pose
        self.shadow_arm_qpos = joint_pos
        self.shadow_gripper = gripper_pos
        return obs

    def _sync_shadow_model(self):
        self.data.qpos[self.base_qpos_indices] = self.shadow_base_pose
        self.data.qpos[self.arm_qpos_indices] = self.shadow_arm_qpos
        mujoco.mj_forward(self.model, self.data)

    def _clip_arm_qpos(self, target_qpos):
        arm_low = []
        arm_high = []
        for name in self.arm_joint_names:
            joint_id = self.model.joint(name).id
            if bool(self.model.jnt_limited[joint_id]):
                arm_low.append(self.model.jnt_range[joint_id][0])
                arm_high.append(self.model.jnt_range[joint_id][1])
            else:
                arm_low.append(-np.inf)
                arm_high.append(np.inf)
        return np.clip(target_qpos, np.array(arm_low), np.array(arm_high)).astype(np.float32)

    def _shadow_ee_world_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy().astype(np.float32)

    def _shadow_ee_world_quat_xyzw(self):
        ee_xmat = self.data.body(self.ee_body_id).xmat.reshape(3, 3)
        quat_xyzw = R.from_matrix(ee_xmat).as_quat()
        return quat_xyzw.astype(np.float32)

    def _shadow_relative_ee_pos(self):
        return self._relative_ee_pos_from_world(self.shadow_base_pose, self._shadow_ee_world_pos())

    def _relative_ee_pos_from_world(self, base_pose, ee_world_pos):
        base_yaw = float(base_pose[2])
        arm_base_xy = np.array(
            [
                base_pose[0] + self.arm_base_offset[0] * np.cos(base_yaw),
                base_pose[1] + self.arm_base_offset[0] * np.sin(base_yaw),
            ],
            dtype=np.float32,
        )
        x, y = relative_position(arm_base_xy, ee_world_pos[:2], base_yaw)
        z = ee_world_pos[2] - self.arm_base_offset[2]
        return np.array([x, y, z], dtype=np.float32)

    def _relative_ee_quat_from_world(self, base_pose, arm_quat_xyzw):
        base_yaw = float(base_pose[2])
        base_quat_wxyz = np.array([np.cos(base_yaw / 2.0), 0.0, 0.0, np.sin(base_yaw / 2.0)], dtype=np.float32)
        arm_quat_wxyz = np.array(
            [arm_quat_xyzw[3], arm_quat_xyzw[0], arm_quat_xyzw[1], arm_quat_xyzw[2]],
            dtype=np.float32,
        )
        rel_wxyz = relative_quaternion(base_quat_wxyz, arm_quat_wxyz).astype(np.float32)
        return rel_wxyz
