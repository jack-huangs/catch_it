"""
Author: Yuanhang Zhang
Version@2024-10-17
All Rights Reserved
ABOUT: this file constains the RL environment for the DCMM task
"""

import os, sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./gym_dcmm/'))
import argparse
import math
import time
print(os.getcwd())
import configs.env.DcmmCfg as DcmmCfg
import cv2 as cv
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM
from gym_dcmm.utils.ik_pkg.ik_base import IKBase
import copy
from termcolor import colored
from decorators import *
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from utils.util import *
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from collections import deque

# os.environ['MUJOCO_GL'] = 'egl'
np.set_printoptions(precision=8)

paused = True
cmd_lin_y = 0.0
cmd_lin_x = 0.0
cmd_ang = 0.0
trigger_delta = False
trigger_delta_hand = False

def env_key_callback(keycode):
  print("chr(keycode): ", (keycode))
  global cmd_lin_y, cmd_lin_x, cmd_ang, paused, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
  if keycode == 265: # AKA: up
    cmd_lin_y += 1
    print("up %f" % cmd_lin_y)
  if keycode == 264: # AKA: down
    cmd_lin_y -= 1
    print("down %f" % cmd_lin_y)
  if keycode == 263: # AKA: left
    cmd_lin_x -= 1
    print("left: %f" % cmd_lin_x)
  if keycode == 262: # AKA: right
    cmd_lin_x += 1
    print("right %f" % cmd_lin_x) 
  if keycode == 52: # AKA: 4
    cmd_ang -= 0.2
    print("turn left %f" % cmd_ang)
  if keycode == 54: # AKA: 6
    cmd_ang += 0.2
    print("turn right %f" % cmd_ang)
  if chr(keycode) == ' ': # AKA: space
    if paused: paused = not paused
  if keycode == 334: # AKA + (on the numpad)
    trigger_delta = True
    delta_xyz = 0.1
  if keycode == 333: # AKA - (on the numpad)
    trigger_delta = True
    delta_xyz = -0.1
  if keycode == 327: # AKA 7 (on the numpad)
    trigger_delta_hand = True
    delta_xyz_hand = 0.2
  if keycode == 329: # AKA 9 (on the numpad)
    trigger_delta_hand = True
    delta_xyz_hand = -0.2

class DcmmVecEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "depth_array", "depth_rgb_array"]}
    """
    Args:
        render_mode: str
            The mode of rendering, including "rgb_array", "depth_array".
        render_per_step: bool
            Whether to render the mujoco model per simulation step.
        viewer: bool
            Whether to show the mujoco viewer.
        imshow_cam: bool
            Whether to show the camera image.
        object_eval: bool
            Use the evaluation object.
        camera_name: str
            The name of the camera.
        object_name: str
            The name of the object.
        env_time: float
            The maximum time of the environment.
        steps_per_policy: int
            The number of steps per action.
        img_size: tuple
            The size of the image.
    """
    def __init__(
        self,
        task="tracking",
        render_mode="depth_array",#在gym环境中，render_mode参数指定了环境渲染的方式。
        render_per_step=False,
        viewer=False,#viewer参数决定是否显示MuJoCo的图形界面，通常在训练过程中会关闭以节省资源，而在调试或演示时会打开。
        viewer_sleep=0.03,#viewer 打开时，每次 sync 后额外 sleep 一小段时间，便于肉眼观察测试过程
        imshow_cam=False,#摄像头的图像
        object_eval=False,#是否使用评估对象（通常是训练过程中未见过的对象），用于测试策略的泛化能力
        camera_name=["top", "wrist"],#
        object_name="object",
        env_time=6.0, #环境的最大时间，单位为秒
        steps_per_policy=20, #每个策略动作持续的仿真步数
        img_size=(480, 640),
        device='cuda:0',
        print_obs=False,
        print_reward=False,
        print_ctrl=False,
        print_info=False,
        print_contacts=False,
    ):
        # 当前项目已经切到 tidybot + Tracking，
        # 这个环境类不再保留旧机器人 / Catching 的运行入口。
        if task != "Tracking":
            raise ValueError("Invalid task: {}".format(task))
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.object_name = object_name
        self.imshow_cam = imshow_cam
        self.task = task
        self.img_size = img_size
        self.device = device
        self.steps_per_policy = steps_per_policy
        self.render_per_step = render_per_step
        self.viewer_sleep = viewer_sleep
        # Print Settings
        self.print_obs = print_obs
        self.print_reward = print_reward
        self.print_ctrl = print_ctrl
        self.print_info = print_info
        self.print_contacts = print_contacts
        # 构建底层 MuJoCo 机器人对象。后面底盘、机械臂、手的控制都由它负责。
        self.Dcmm = MJ_DCMM(viewer=viewer, object_name=object_name, object_eval=object_eval)
        self.use_tidybot = self.Dcmm.use_tidybot
        if not self.use_tidybot:
            raise ValueError("DcmmVecEnv 现在只支持 tidybot 模型。")
        # self.Dcmm.show_model_info()
        # fps 不是渲染帧率，而是“策略步”对应的观测差分频率
        self.policy_dt = self.steps_per_policy * self.Dcmm.model.opt.timestep
        self.fps = 1 / self.policy_dt
        self.base_vel_lpf_alpha = self.policy_dt / (0.2 + self.policy_dt)
        # 下面这些变量控制每个 episode 中物体的随机化与抛掷逻辑
        self.random_mass = 0.25
        self.object_static_time = 0.75
        self.object_throw = False #物体是否已经被扔出
        self.object_train = True
        if object_eval: self.set_object_eval()
        # 根据训练/评估模式重写 XML 中的物体参数，然后重新创建 MuJoCo model/data
        self.Dcmm.model_xml_string = self._reset_object()
        self.Dcmm.model = mujoco.MjModel.from_xml_string(self.Dcmm.model_xml_string)
        self.Dcmm.data = mujoco.MjData(self.Dcmm.model)
        # tidybot 重新载入 model 后，要同步刷新各类索引和默认姿态
        self.Dcmm.arm_qpos_indices = np.array([self.Dcmm.model.joint(name).qposadr[0] for name in self.Dcmm.arm_joint_names], dtype=int)
        self.Dcmm.hand_qpos_indices = np.array([self.Dcmm.model.joint(name).qposadr[0] for name in self.Dcmm.hand_joint_names], dtype=int)
        self.Dcmm.pad_geom_ids = np.array(
            [mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in DcmmCfg.pad_geom_names],
            dtype=int,
        )
        base_body = self.Dcmm.model.body(self.Dcmm.base_body_name)
        start = int(base_body.geomadr[0])
        count = int(base_body.geomnum[0])
        self.Dcmm.base_geom_ids = np.arange(start, start + count, dtype=int)
        self.Dcmm.data.qpos[self.Dcmm.arm_qpos_indices] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.qpos[self.Dcmm.hand_qpos_indices] = DcmmCfg.hand_joints[:]
        self.Dcmm.target_base_pose[:] = self.Dcmm.data.qpos[0:3]
        mujoco.mj_forward(self.Dcmm.model, self.Dcmm.data)
        # 提前记录关键 geom 的 id，后面做接触检测时会反复使用
        self.hand_start_id = -1
        self.pad_geom_ids = self.Dcmm.pad_geom_ids.copy()
        self.floor_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.object_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)
        self.base_id = -1

        # 配置相机和渲染器
        self.Dcmm.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.Dcmm.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.mujoco_renderer = MujocoRenderer(
            self.Dcmm.model, self.Dcmm.data
        )
        if self.Dcmm.open_viewer:
            if self.Dcmm.viewer:
                print("Close the previous viewer")
                self.Dcmm.viewer.close()
            self.Dcmm.viewer = mujoco.viewer.launch_passive(self.Dcmm.model, self.Dcmm.data, key_callback=env_key_callback)
            # Modify the view position and orientation
            self.Dcmm.viewer.cam.lookat[0:2] = [0, 1]
            self.Dcmm.viewer.cam.distance = 5.0
            self.Dcmm.viewer.cam.azimuth = 180
            # self.viewer.cam.elevation = -1.57
        else: self.Dcmm.viewer = None

        # 观测空间：只保留 tidybot Tracking 所需状态
        arm_joint_names = self.Dcmm.arm_joint_names
        arm_low_obs = []
        arm_high_obs = []
        for name in arm_joint_names:
            joint_id = self.Dcmm.model.joint(name).id
            if bool(self.Dcmm.model.jnt_limited[joint_id]):
                arm_low_obs.append(self.Dcmm.model.jnt_range[joint_id][0])
                arm_high_obs.append(self.Dcmm.model.jnt_range[joint_id][1])
            else:
                arm_low_obs.append(-np.pi)
                arm_high_obs.append(np.pi)
        arm_low_obs = np.array(arm_low_obs, dtype=np.float32)
        arm_high_obs = np.array(arm_high_obs, dtype=np.float32)
        hand_low_obs = np.array([0.0], dtype=np.float32)
        hand_high_obs = np.array([255.0], dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "base": spaces.Dict({
                    "v_lin_2d": spaces.Box(-4, 4, shape=(2,), dtype=np.float32),#二维线速度，BOX连续空间，范围是-4到4，形状是2维，数据类型是float32
                }),
                "arm": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),#end-effector 3D position	末端三维位置
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),#end-effector  quaternion	末端姿态四元数
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),#end-effector 3D linear velocity	末端三维线速度
                    "joint_pos": spaces.Box(low = arm_low_obs,
                                            high = arm_high_obs,
                                            dtype=np.float32),
                }),
                "hand": spaces.Box(low = hand_low_obs,
                                   high = hand_high_obs,
                                   dtype=np.float32),
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),
                    ## TODO: to be determined
                    # "shape": spaces.Box(-5, 5, shape=(2,), dtype=np.float32),
                }),
            }
        )
        # Define the limit for the mobile base action
        base_low = np.array([-4, -4])
        base_high = np.array([4, 4])
        # Define the limit for the arm action
        arm_low = -0.025*np.ones(7)
        arm_high = 0.025*np.ones(7)
        # Tracking 不主动控制夹爪，这里保留 1 维占位动作，默认都会被置零
        hand_low = np.array([-1.0], dtype=np.float32)
        hand_high = np.array([1.0], dtype=np.float32)

        # Get initial ee_pos3d
        self.init_pos = True
        self.initial_ee_pos3d = self._get_relative_ee_pos3d()
        self.initial_obj_pos3d = self._get_relative_object_pos3d()
        self.prev_ee_pos3d = np.array([0.0, 0.0, 0.0]) #上一步末端位置
        self.prev_obj_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:] 
        self.prev_obj_pos3d[:] = self.initial_obj_pos3d[:]

        # 动作空间也是字典：
        # base 控底盘速度，arm 控末端位姿增量，hand 控手指关节
        # 每个low，high表示动作或状态每一维允许的最小值和最大值。
        self.action_space = spaces.Dict(
            {
                "base": spaces.Box(base_low, base_high, shape=(2,), dtype=np.float32),
                "arm": spaces.Box(arm_low, arm_high, shape=arm_low.shape, dtype=np.float32),
                "hand": spaces.Box(low = hand_low,
                                   high = hand_high,
                                   dtype = np.float32),
            }
        )
        # 动作延迟缓冲区：模拟真实系统中“策略输出后不会立刻生效”的情况
        self.action_buffer = {
            "base": DynamicDelayBuffer(maxlen=2),
            "arm": DynamicDelayBuffer(maxlen=2),
            "hand": DynamicDelayBuffer(maxlen=2),
        }
        # Combine the limits of the action space
        self.actions_low = np.concatenate([base_low, arm_low, hand_low])
        self.actions_high = np.concatenate([base_high, arm_high, hand_high])

        self.obs_dim = get_total_dimension(self.observation_space)
        self.act_dim = get_total_dimension(self.action_space)
        # 新机器人改成 7 关节 Tracking，关节位置对策略很重要，所以不再从 Tracking 观测里删掉 arm joint_pos
        self.obs_t_dim = self.obs_dim - 1
        self.act_t_dim = self.act_dim - 1
        self.obs_c_dim = self.obs_dim
        self.act_c_dim = self.act_dim
        print("##### Tracking Task \n obs_dim: {}, act_dim: {}".format(self.obs_t_dim, self.act_t_dim))

        # 环境运行过程中的状态变量
        self.arm_limit = True 
        self.terminated = False
        self.start_time = self.Dcmm.data.time
        self.catch_time = self.Dcmm.data.time - self.start_time
        self.reward_touch = 0
        self.reward_stability = 0 
        self.env_time = env_time
        self.stage_list = ["tracking", "grasping"]
        # Default stage is "tracking"
        self.stage = self.stage_list[0] #当前阶段是 tracking 还是 grasping
        self.steps = 0

        self.prev_ctrl = np.zeros(self.act_dim)
        self.init_ctrl = True
        self.vel_init = False
        self.vel_history = deque(maxlen=4)

        self.info = {
            "ee_distance": np.linalg.norm(self._ee_world_pos() -
                                          self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "base_distance": np.linalg.norm(self.Dcmm.data.body(self.Dcmm.arm_base_body_name).xpos[0:2] -
                                            self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "env_time": self.Dcmm.data.time - self.start_time,
            "imgs": {}
        }
        self.contacts = {
            # Get contact point from the mujoco model
            "object_contacts": np.array([]),
            "hand_contacts": np.array([]),
        }

        self.object_q = np.array([1, 0, 0, 0])
        self.object_pos3d = np.array([0, 0, 1.5])
        self.object_vel6d = np.array([0., 0., 1.25, 0.0, 0.0, 0.0])
        self.step_touch = False

        self.imgs = np.zeros((0, self.img_size[0], self.img_size[1], 1))

        # Random PID Params
        self.k_arm = np.ones(len(DcmmCfg.arm_joints))
        self.k_drive = np.ones(2)
        self.k_steer = np.ones(1)
        self.k_hand = np.ones(1)
        # Random Obs & Act Params
        self.k_obs_base = DcmmCfg.k_obs_base
        self.k_obs_arm = DcmmCfg.k_obs_arm
        self.k_obs_hand = DcmmCfg.k_obs_hand
        self.k_obs_object = DcmmCfg.k_obs_object
        self.k_act = DcmmCfg.k_act

    def set_object_eval(self):
        self.object_train = False

    def update_render_state(self, render_per_step):
        self.render_per_step = render_per_step

    def update_stage(self, stage):
        if stage in self.stage_list:
            self.stage = stage
        else:
            raise ValueError("Invalid stage: {}".format(stage))

    def _get_contacts(self):
        # 只保留 tidybot Tracking 的接触逻辑：
        # hand = 夹爪 pad geom，base = 底盘 geom，object = 目标物体 geom
        geom_ids = self.Dcmm.data.contact.geom
        geom1_ids = self.Dcmm.data.contact.geom1
        geom2_ids = self.Dcmm.data.contact.geom2
        geom1_hand = np.where(np.isin(geom1_ids, self.pad_geom_ids))[0]
        geom2_hand = np.where(np.isin(geom2_ids, self.pad_geom_ids))[0]
        ## get the contact points of the hand
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_hand.size != 0:
            contacts_geom1 = geom_ids[geom1_hand][:,1]
        if geom2_hand.size != 0:
            contacts_geom2 = geom_ids[geom2_hand][:,0]
        hand_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        ## get the contact points of the object
        geom1_object = np.where((geom1_ids == self.object_id))[0]
        geom2_object = np.where((geom2_ids == self.object_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_object.size != 0:
            contacts_geom1 = geom_ids[geom1_object][:,1]
        if geom2_object.size != 0:
            contacts_geom2 = geom_ids[geom2_object][:,0]
        object_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        ## get the contact points of the base
        geom1_base = np.where(np.isin(geom1_ids, self.Dcmm.base_geom_ids))[0]
        geom2_base = np.where(np.isin(geom2_ids, self.Dcmm.base_geom_ids))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_base.size != 0:
            contacts_geom1 = geom_ids[geom1_base][:,1]
        if geom2_base.size != 0:
            contacts_geom2 = geom_ids[geom2_base][:,0]
        base_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        if self.print_contacts:
            print("object_contacts: ", object_contacts)
            print("hand_contacts: ", hand_contacts)
            print("base_contacts: ", base_contacts)
        return {
            # Get contact point from the mujoco model
            "object_contacts": object_contacts,
            "hand_contacts": hand_contacts,
            "base_contacts": base_contacts
        }

    def _ee_world_pos(self):
        # tidybot 用 pinch_site 作为夹爪末端位置
        return self.Dcmm.data.site(self.Dcmm.ee_site_name).xpos.copy()

    def _get_base_vel(self):
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        global_base_vel = self.Dcmm.data.qvel[0:2]
        base_vel_x = math.cos(base_yaw) * global_base_vel[0] + math.sin(base_yaw) * global_base_vel[1]
        base_vel_y = -math.sin(base_yaw) * global_base_vel[0] + math.cos(base_yaw) * global_base_vel[1]
        return np.array([base_vel_x, base_vel_y])

    def _get_relative_ee_pos3d(self):
        # Caclulate the ee_pos3d w.r.t. the base_link
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        ee_pos = self._ee_world_pos()
        x,y = relative_position(self.Dcmm.data.body(self.Dcmm.arm_base_body_name).xpos[0:2], 
                                ee_pos[0:2], 
                                base_yaw)
        return np.array([x, y, 
                         ee_pos[2]-self.Dcmm.data.body(self.Dcmm.arm_base_body_name).xpos[2]])

    def _get_relative_ee_quat(self):
        # Caclulate the ee_pos3d w.r.t. the base_link
        quat = relative_quaternion(self.Dcmm.data.body("base_link").xquat, self.Dcmm.data.body(self.Dcmm.ee_body_name).xquat)
        return np.array(quat)

    def _get_relative_ee_v_lin_3d(self):
        # Caclulate the ee_v_lin3d w.r.t. the base_link
        # In simulation, we can directly get the velocity of the end-effector
        base_vel = self.Dcmm.data.body(self.Dcmm.arm_base_body_name).cvel[3:6]
        global_ee_v_lin = self.Dcmm.data.body(self.Dcmm.ee_body_name).cvel[3:6]
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        ee_v_lin_x = math.cos(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        ee_v_lin_y = -math.sin(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        # TODO: In the real world, we can only estimate it by differentiating the position
        return np.array([ee_v_lin_x, ee_v_lin_y, global_ee_v_lin[2]-base_vel[2]])
    
    def _get_relative_object_pos3d(self):
        # Caclulate the object_pos3d w.r.t. the base_link
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        x,y = relative_position(self.Dcmm.data.body(self.Dcmm.arm_base_body_name).xpos[0:2], 
                                self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2], 
                                base_yaw)
        return np.array([x, y, 
                         self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body(self.Dcmm.arm_base_body_name).xpos[2]])

    def _get_relative_object_v_lin_3d(self):
        # Caclulate the object_v_lin3d w.r.t. the base_link
        base_vel = self.Dcmm.data.body(self.Dcmm.arm_base_body_name).cvel[3:6]
        global_object_v_lin = self.Dcmm.data.joint(self.Dcmm.object_name).qvel[0:3]
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        object_v_lin_x = math.cos(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_object_v_lin[1]-base_vel[1])
        object_v_lin_y = -math.sin(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_object_v_lin[1]-base_vel[1])
        return np.array([object_v_lin_x, object_v_lin_y, global_object_v_lin[2]-base_vel[2]])

    def _get_obs(self):
        # 这里不是“执行动作”的地方，而是“动作执行完以后，把新的物理状态读出来”
        # step() 里先调用 _step_mujoco_simulation(action) 执行动作，
        # MuJoCo 更新完机器人和物体状态后，再由 _get_obs() 把“新状态”整理成观测

        # 读取末端和物体在相对坐标系下的位置
        # 这些值已经反映了“刚才动作执行之后”的最新状态
        ee_pos3d = self._get_relative_ee_pos3d()
        obj_pos3d = self._get_relative_object_pos3d()

        # reset 后第一次取观测时，还没有“上一帧位置”
        # 所以先把 prev_* 初始化成当前值，避免后面速度差分异常
        if self.init_pos:
            self.prev_ee_pos3d[:] = ee_pos3d[:]
            self.prev_obj_pos3d[:] = obj_pos3d[:]
            self.init_pos = False

        # 把当前时刻的机器人/物体状态整理成策略网络要看的 obs 字典
        # 同时加入少量高斯噪声，模拟真实传感器误差，提高训练鲁棒性
        obs = {
            "base": {
                # 底盘二维线速度
                "v_lin_2d": self._get_base_vel() + np.random.normal(0, self.k_obs_base, 2),
            },
            "arm": {
                # 机械臂末端位置（相对坐标）
                "ee_pos3d": ee_pos3d + np.random.normal(0, self.k_obs_arm, 3),
                # 机械臂末端姿态四元数（相对坐标）
                "ee_quat": self._get_relative_ee_quat() + np.random.normal(0, self.k_obs_arm, 4),
                # 末端线速度这里不是直接读传感器，而是用“当前位置 - 上一帧位置”做差分近似
                # 所以它表示：刚才动作执行后，末端移动得有多快
                'ee_v_lin_3d': (ee_pos3d - self.prev_ee_pos3d)*self.fps + np.random.normal(0, self.k_obs_arm, 3),
                # 机械臂 6 个关节角
                "joint_pos": np.array(self.Dcmm.data.qpos[self.Dcmm.arm_qpos_indices]) + np.random.normal(0, self.k_obs_arm, len(self.Dcmm.arm_qpos_indices)),
            },
            # 手部观测
            "hand": self._get_hand_obs() + np.random.normal(0, self.k_obs_hand, len(self._get_hand_obs())),
            "object": {
                # 物体位置（相对坐标）
                "pos3d": obj_pos3d + np.random.normal(0, self.k_obs_object, 3),
                # "v_lin_3d": self._get_relative_object_v_lin_3d() + np.random.normal(0, self.k_obs_object, 3),
                # 物体速度同样用相邻两帧的位置差分近似
                # 所以它也反映了动作执行后、物体在这一小段时间里的运动变化
                "v_lin_3d": (obj_pos3d - self.prev_obj_pos3d)*self.fps + np.random.normal(0, self.k_obs_object, 3),
            },
        }
        # 当前帧观测组装完后，把当前位置保存起来，供下一步差分速度使用
        self.prev_ee_pos3d = ee_pos3d
        self.prev_obj_pos3d = obj_pos3d
        if self.print_obs:
            print("##### print obs: \n", obs)
        return obs
        # return obs_tensor

    def _get_hand_obs(self):
        # 夹爪当前开合状态，这里只取右侧 driver joint 的位置作 1 维观测
        return np.array([self.Dcmm.data.qpos[self.Dcmm.hand_qpos_indices[0]]], dtype=np.float32)
    
    def _get_info(self):
        # Time of the Mujoco environment
        env_time = self.Dcmm.data.time - self.start_time
        ee_distance = np.linalg.norm(self._ee_world_pos() - 
                                    self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3])
        base_distance = np.linalg.norm(self.Dcmm.data.body(self.Dcmm.arm_base_body_name).xpos[0:2] -
                                        self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2])
        # print("base_distance: ", base_distance)
        if self.print_info: 
            print("##### print info")
            print("env_time: ", env_time)
            print("ee_distance: ", ee_distance)
        return {
            # Get contact point from the mujoco model
            "env_time": env_time,
            "ee_distance": ee_distance,
            "base_distance": base_distance,
        }
    
    def update_target_ctrl(self):
        # 把“这一步策略输出的目标动作”压进延迟缓冲区
        self.action_buffer["base"].append(copy.deepcopy(self.Dcmm.target_base_vel[:]))
        self.action_buffer["arm"].append(copy.deepcopy(self.Dcmm.target_arm_qpos[:]))
        self.action_buffer["hand"].append(copy.deepcopy(self.Dcmm.target_hand_qpos[:]))

    def _get_ctrl(self):
        # tidybot 的 actuator 本身就是 position controller；
        # 这里直接把“目标底盘位置 + 目标关节角 + 目标夹爪控制量”写成 ctrl
        ctrl = np.zeros(self.Dcmm.model.nu)
        self.Dcmm.move_base_vel(self.action_buffer["base"][0])
        ctrl[0:2] = self.Dcmm.target_base_pose[0:2]
        ctrl[2] = self.Dcmm.target_base_pose[2]
        ctrl[3:10] = self.action_buffer["arm"][0]
        ctrl[10] = self.action_buffer["hand"][0][0]
        # Tracking 训练里希望 tidybot 尽量平稳；只有显式配置了执行噪声时才扰动控制量。
        if self.k_act > 0:
            ctrl *= np.random.normal(1, self.k_act, ctrl.shape[0])
        return ctrl

    def _reset_object(self):
        # Parse the XML string
        root = ET.fromstring(self.Dcmm.model_xml_string)

        # Find the <body> element with name="object"
        object_body = root.find(".//body[@name='object']")
        if object_body is not None:
            inertial = object_body.find("inertial")
            if inertial is not None:
                # Generate a random mass within the specified range
                self.random_mass = np.random.uniform(DcmmCfg.object_mass[0], DcmmCfg.object_mass[0])
                # Update the mass attribute
                inertial.set("mass", str(self.random_mass))
            joint = object_body.find("joint")
            if joint is not None:
                # Generate a random damping within the specified range
                random_damping = np.random.uniform(DcmmCfg.object_damping[0], DcmmCfg.object_damping[1])
                # Update the damping attribute
                joint.set("damping", str(random_damping))
            # Find the <geom> element
            geom = object_body.find(".//geom[@name='object']")
            if geom is not None:
                # Modify the type and size attributes
                object_id = np.random.choice([0, 1, 2, 3, 4])
                if self.object_train:
                    object_shape = DcmmCfg.object_shape[object_id]
                    geom.set("type", object_shape)  # Replace "box" with the desired type
                    object_size = np.array([np.random.uniform(low=low, high=high) for low, high in DcmmCfg.object_size[object_shape]])
                    geom.set("size", np.array_str(object_size)[1:-1])  # Replace with the desired size
                    # print("### Object Geom Info ###")
                    # for key, value in geom.attrib.items():
                    #     print(f"{key}: {value}")
                else:
                    object_mesh = DcmmCfg.object_mesh[object_id]
                    geom.set("mesh", object_mesh)
        # Convert the XML element tree to a string
        xml_str = ET.tostring(root, encoding='unicode')
        
        return xml_str

    def random_object_pose(self):
        # 让物体主要从 tidybot 正前方飞来：
        # 位置集中在 +Y 方向，速度主分量沿 -Y 指向机器人，只保留少量左右扰动。
        x = np.random.uniform(*DcmmCfg.tracking_object_x_range)
        y = np.random.uniform(*DcmmCfg.tracking_object_y_range)
        low_factor = np.random.rand() < 0.5
        if low_factor:
            height = np.random.uniform(*DcmmCfg.tracking_object_low_height)
        else:
            height = np.random.uniform(*DcmmCfg.tracking_object_high_height)
        v_lin_x = np.random.uniform(-DcmmCfg.tracking_object_lateral_speed,
                                    DcmmCfg.tracking_object_lateral_speed)
        v_lin_y = -np.random.uniform(*DcmmCfg.tracking_object_forward_speed)
        v_lin_z = np.random.uniform(*DcmmCfg.tracking_object_vertical_speed)
        if y > 2.3:
            v_lin_y -= 0.2
        if height < 1.0:
            v_lin_z += 0.4
        self.object_pos3d = np.array([x, y, height])
        self.object_vel6d = np.array([v_lin_x, v_lin_y, v_lin_z, 0.0, 0.0, 0.0])
        # Random Static Time
        self.object_static_time = np.random.uniform(DcmmCfg.object_static[0], DcmmCfg.object_static[1])
        # Random Quaternion
        r_obj_quat = R.from_euler('xyz', [0, np.random.rand()*1*math.pi, 0], degrees=False)
        self.object_q = r_obj_quat.as_quat()

    
    def random_PID(self):
        if self.Dcmm.use_tidybot:
            # tidybot 当前直接使用 position actuator 追踪目标关节角 / 底盘位置，
            # 这里不再随机化 PID 或延迟缓冲，避免训练和手动调试时出现无意义抖动。
            self.k_arm = np.ones(len(DcmmCfg.arm_joints))
            self.k_drive = np.ones(self.k_drive.shape[0])
            self.k_steer = np.ones(self.k_steer.shape[0])
            self.k_hand = np.ones(1)
            self.action_buffer["base"].set_maxlen(1)
            self.action_buffer["arm"].set_maxlen(1)
            self.action_buffer["hand"].set_maxlen(1)
            self.action_buffer["base"].clear()
            self.action_buffer["arm"].clear()
            self.action_buffer["hand"].clear()
            return
        # Random the PID Controller Params in DCMM
        self.k_arm = np.random.uniform(0, 1, size=len(DcmmCfg.arm_joints))
        self.k_drive = np.random.uniform(0, 1, size=self.k_drive.shape[0])
        self.k_steer = np.random.uniform(0, 1, size=self.k_steer.shape[0])
        self.k_hand = np.random.uniform(0, 1, size=1)
        # Reset the PID Controller
        self.Dcmm.arm_pid.reset(self.k_arm*(DcmmCfg.k_arm[1]-DcmmCfg.k_arm[0])+DcmmCfg.k_arm[0])#控机械臂的 PID，作用是把“策略输出的末端位姿增量”转成“每个机械臂关节的目标位置”
        self.Dcmm.hand_pid.reset(self.k_hand[0]*(DcmmCfg.k_hand[1]-DcmmCfg.k_hand[0])+DcmmCfg.k_hand[0])#控手指关节的 PID，作用是把“策略输出的手指关节位置”转成“每个手指关节的力
        # Random the Delay Buffer Params in DCMM
        self.action_buffer["base"].set_maxlen(np.random.choice(DcmmCfg.act_delay['base']))
        self.action_buffer["arm"].set_maxlen(np.random.choice(DcmmCfg.act_delay['arm']))
        self.action_buffer["hand"].set_maxlen(np.random.choice(DcmmCfg.act_delay['hand']))
        # Clear Buffer
        self.action_buffer["base"].clear()
        self.action_buffer["arm"].clear()
        self.action_buffer["hand"].clear()

    def _reset_simulation(self):
        # 真正的“物理世界重置”都在这里做：
        # 清空状态、还原关节、随机目标物体、随机重力、随机 PID、随机延迟
        mujoco.mj_resetData(self.Dcmm.model, self.Dcmm.data)
        if self.Dcmm.data_arm is not None:
            mujoco.mj_resetData(self.Dcmm.model_arm, self.Dcmm.data_arm)
        if self.Dcmm.model.na == 0:
            self.Dcmm.data.act[:] = None
        if self.Dcmm.data_arm is not None and self.Dcmm.model_arm.na == 0:
            self.Dcmm.data_arm.act[:] = None
        self.Dcmm.data.ctrl = np.zeros(self.Dcmm.model.nu)
        if self.Dcmm.data_arm is not None:
            self.Dcmm.data_arm.ctrl = np.zeros(self.Dcmm.model_arm.nu)
        # 把底盘直接放到“正面对球”的初始位姿。
        self.Dcmm.data.qpos[0:3] = DcmmCfg.base_init_pose[:]
        self.Dcmm.data.qpos[self.Dcmm.arm_qpos_indices] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.qpos[self.Dcmm.hand_qpos_indices] = DcmmCfg.hand_joints[:]
        if self.Dcmm.data_arm is not None:
            self.Dcmm.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:6]
        self.Dcmm.data.body("object").xpos[0:3] = np.array([2, 2, 1])
        # 随机生成物体初始位置、初速度和姿态
        self.random_object_pose()
        self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                    velocity=np.zeros(6))
        # TODO: TESTING
        # self.Dcmm.set_throw_pos_vel(pose=np.array([0.0, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0]),
        #                             velocity=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        # tidybot Tracking 先固定重力，避免底盘和机械臂在同一条策略上还要适配随机重力，
        # 这会放大 position actuator 的抖动感；object 的随机抛掷仍然保留。
        if self.Dcmm.use_tidybot:
            self.Dcmm.model.opt.gravity[2] = -9.81
        else:
            self.Dcmm.model.opt.gravity[2] = -9.81 + 0.5*np.random.uniform(-1, 1)
        # Random PID
        self.random_PID()
        # Forward Kinematics
        mujoco.mj_forward(self.Dcmm.model, self.Dcmm.data)
        if self.Dcmm.data_arm is not None:
            mujoco.mj_forward(self.Dcmm.model_arm, self.Dcmm.data_arm)


    def reset(self):
        # Gym 接口：开始一个新 episode
        # 这里会重置物理世界、重置任务状态，并返回初始 observation / info
        self._reset_simulation()
        self.init_ctrl = True
        self.init_pos = True
        self.vel_init = False
        self.object_throw = False
        self.steps = 0
        # Reset the time
        self.start_time = self.Dcmm.data.time
        self.catch_time = self.Dcmm.data.time - self.start_time

        ## Reset the target velocity of the mobile base
        self.Dcmm.target_base_vel = np.array([0.0, 0.0, 0.0])
        self.Dcmm.target_base_pose[:] = self.Dcmm.data.qpos[0:3]
        ## Reset the target joint positions of the arm
        self.Dcmm.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        ## Reset the target joint positions of the hand
        self.Dcmm.target_hand_qpos[:] = DcmmCfg.hand_joints[:]
        ## Reset the reward
        self.stage = "tracking"
        self.terminated = False
        self.reward_touch = 0
        self.reward_stability = 0

        self.info = {
            "ee_distance": np.linalg.norm(self._ee_world_pos() - 
                                       self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "base_distance": np.linalg.norm(self.Dcmm.data.body(self.Dcmm.arm_base_body_name).xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "evn_time": self.Dcmm.data.time - self.start_time,
        }
        # 重新计算本回合起点的观测和辅助信息
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d = self._get_relative_object_pos3d()
        observation = self._get_obs()
        info = self._get_info()
        # 初始时刻也可以返回一帧渲染结果，供可视化或调试使用
        imgs = self.render()
        info['imgs'] = imgs
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))

        return observation, info

    def norm_ctrl(self, ctrl, components): #计算控制量的大小，超参数 r_ctrl
        '''
        Convert the ctrl (dict type) to the numpy array and return its norm value
        Input: ctrl, dict
        Return: norm, float
        '''
        ctrl_array = np.concatenate([
            np.asarray(ctrl[component], dtype=np.float32) * DcmmCfg.reward_weights['r_ctrl'][component]
            for component in components
        ])
        return np.linalg.norm(ctrl_array)


    def compute_reward(self, obs, info, ctrl):
        '''
        Rewards:
        - Object Position Reward
        - Object Orientation Reward
        - Object Touch Success Reward
        - Object Catch Stability Reward
        - Collision Penalty
        - Constraint Penalty
        '''
        # 这个函数就是“裁判”：
        # 根据机器人本步的表现给出一个标量奖励 reward
        rewards = 0.0
        # 距离相关奖励：
        # 如果这一步比上一步更接近物体，就得到正奖励
        #(1) 底座接近奖励
        reward_base_pos = (self.info["base_distance"] - info["base_distance"]) * DcmmCfg.reward_weights["r_base_pos"] #self.info表示“上一步保存下来的信息”
        #(2) 末端接近奖励
        reward_ee_pos = (self.info["ee_distance"] - info["ee_distance"]) * DcmmCfg.reward_weights["r_ee_pos"]
        #(3) 精确接近奖励，只有当末端非常接近物体时，这项才会大
        reward_ee_precision = math.exp(-50*info["ee_distance"]**2) * DcmmCfg.reward_weights["r_precision"]

        # 碰撞惩罚：如果底盘/机器人发生了不该有的接触，就扣分
        reward_collision = 0
        if self.contacts['base_contacts'].size != 0:
            reward_collision = DcmmCfg.reward_weights["r_collision"] #如果发生了碰撞
        
        # (5) 约束惩罚：IK 失败或机械臂越界时，说明动作不合理，也要扣分
        reward_constraint = 0 if self.arm_limit else -1
        reward_constraint *= DcmmCfg.reward_weights["r_constraint"]

        # (6) 接触奖励：本步如果成功碰到物体，就给额外奖励
        if self.step_touch:
            print("TRACK SUCCESS!!!!!")
            if not self.reward_touch:
                self.catch_time = self.Dcmm.data.time - self.start_time
            self.reward_touch = DcmmCfg.reward_weights["r_touch"][self.task]
        else:
            self.reward_touch = 0

        # Tracking 任务只关心“把夹爪末端追到物体附近”
        reward_ctrl = - self.norm_ctrl(ctrl, {"base", "arm"})
        if info["ee_distance"] < DcmmCfg.tracking_close_bonus_thresh:
            reward_close = (1.0 - info["ee_distance"] / DcmmCfg.tracking_close_bonus_thresh) \
                * DcmmCfg.reward_weights["r_close"]
        else:
            reward_close = 0.0
        # tidybot 的 Tracking 更希望“夹爪末端朝向”和“飞来物体速度方向”一致
        rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
        gripper_forward_axis = rotation_matrix @ np.array([0.0, 0.0, -1.0])
        object_velocity = obs["object"]["v_lin_3d"]
        reward_orient = max(cos_angle_between_vectors(object_velocity, gripper_forward_axis), 0.0) \
            * DcmmCfg.reward_weights["r_orient"]
        rewards = reward_base_pos + reward_ee_pos + reward_ee_precision + reward_close + reward_orient + reward_ctrl + reward_collision + reward_constraint + self.reward_touch
        if self.print_reward:
            if reward_constraint < 0:
                print("ctrl: ", ctrl)
            print("### print reward")
            print("reward_ee_pos: {:.3f}, reward_ee_precision: {:.3f}, reward_close: {:.3f}, reward_orient: {:.3f}, reward_ctrl: {:.3f}, \n".format(
                reward_ee_pos, reward_ee_precision, reward_close, reward_orient, reward_ctrl
            ) + "reward_collision: {:.3f}, reward_constraint: {:.3f}, reward_touch: {:.3f}".format(
                reward_collision, reward_constraint, self.reward_touch
            ))
            print("total reward: {:.3f}\n".format(rewards))
        
        return rewards

    def _step_mujoco_simulation(self, action_dict):
        # 这个函数是真正的“执行动作”：
        # 它把 PPO 给的高层动作转成机器人目标，再在 MuJoCo 里连续模拟若干小步
        # gym 单环境调试时，action_dict 里的值可能还是 Python list；这里先统一转成 ndarray
        action_dict = {k: np.asarray(v, dtype=np.float32) for k, v in action_dict.items()}
        self.Dcmm.target_base_vel[0:2] *= 1.0 - self.base_vel_lpf_alpha
        self.Dcmm.target_base_vel[0:2] += self.base_vel_lpf_alpha * action_dict["base"]
        # tidybot Tracking 直接输出 7 关节增量，不再走旧的末端 IK
        _, success = self.Dcmm.set_arm_target_qpos(action_dict["arm"])
        self.arm_limit = success
        # 夹爪在 Tracking 里默认保持张开；这里仍保留 1 维占位接口，方便以后需要时细调
        self.Dcmm.action_hand2qpos(action_dict["hand"])
        # 把目标动作压入延迟缓冲区，后续控制器读取缓冲区前端的动作执行
        self.update_target_ctrl()
        # 每一步开始前先清空“本步是否碰到物体”的标志
        self.step_touch = False
        # 一次策略动作会执行 steps_per_policy 个 MuJoCo 小步
        for _ in range(self.steps_per_policy):
            # 根据目标动作和当前状态，生成底层控制量并写入 MuJoCo
            self.Dcmm.data.ctrl[:] = self._get_ctrl()
            if self.render_per_step:
                # Rendering
                img = self.render()
            # As of MuJoCo 2.0, force-related quantities like cacc are not computed
            # unless there's a force sensor in the model.
            # See https://github.com/openai/gym/issues/1541
            # 物体一开始会静止悬一小段时间，然后才真正被“扔”出去
            if self.Dcmm.data.time - self.start_time < self.object_static_time:
                self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                            velocity=np.zeros(6))
            elif not self.object_throw:
                self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                            velocity=self.object_vel6d[:])
                self.object_throw = True

            mujoco.mj_step(self.Dcmm.model, self.Dcmm.data)
            mujoco.mj_rnePostConstraint(self.Dcmm.model, self.Dcmm.data)

            # 更新接触信息，用于后面判断“成功触碰”还是“失败碰撞”
            self.contacts = self._get_contacts()
            # 底盘碰撞直接视为失败
            if self.contacts['base_contacts'].size != 0:
                self.terminated = True
            object_contacts = self.contacts['object_contacts']
            mask_pad = np.isin(object_contacts, self.pad_geom_ids)
            mask_coll = np.logical_not(mask_pad)
            mask_finger = mask_pad
            mask_palm = mask_pad
            # 根据接触部位判断是否算“成功碰到物体”
            if self.step_touch == False and (np.any(mask_palm) or np.any(mask_finger)):
                self.step_touch = True
            # 根据错误接触判断是否提前失败结束
            if not self.terminated:
                # Tracking 放宽后，finger 不再算失败，只有真正错误碰撞才算失败
                self.terminated = np.any(mask_coll)
            # 一旦失败，本步后面的 MuJoCo 小步就不用再跑了
            if self.terminated:
                break

    def step(self, action):
        # Gym 接口：策略每给一次动作，就调用一次 step()
        # 这里会返回“下一状态、奖励、是否结束、额外信息”
        self.steps += 1
        self._step_mujoco_simulation(action)
        # 先根据最新仿真状态重新计算观测和辅助信息
        obs = self._get_obs()
        info = self._get_info()
        if info['ee_distance'] < DcmmCfg.tracking_success_thresh:
            # 对 tidybot 的 Tracking 放宽成功判定：
            # 只要夹爪末端已经非常接近物体，就直接视为成功，
            # 不再强依赖 pad 几何体发生离散接触。
            self.step_touch = True
        # 根据本步表现打分
        reward = self.compute_reward(obs, info, action)
        self.info["base_distance"] = info["base_distance"]
        self.info["ee_distance"] = info["ee_distance"]
        # 附带一帧图像信息，供调试/可视化使用
        imgs = self.render()
        # Update the imgs
        info['imgs'] = imgs
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))
        # terminated：失败结束（碰撞、错误接触等）
        # truncated：正常结束（Tracking 碰到目标）
        truncated = bool(self.step_touch)
        terminated = self.terminated
        done = terminated or truncated
        if done:
            # TEST ONLY
            # self.reset()
            pass
        return obs, reward, terminated, truncated, info

    def preprocess_depth_with_mask(self, rgb_img, depth_img, 
                                   depth_threshold=3.0, 
                                   num_white_points_range=(5, 15),
                                   point_size_range=(1, 5)):
        # Define RGB Filter
        lower_rgb = np.array([5, 0, 0])
        upper_rgb = np.array([255, 15, 15])
        rgb_mask = cv.inRange(rgb_img, lower_rgb, upper_rgb)
        depth_mask = cv.inRange(depth_img, 0, depth_threshold)
        combined_mask = np.logical_and(rgb_mask, depth_mask)
        # Apply combined mask to depth image
        masked_depth_img = np.where(combined_mask, depth_img, 0)
        # Calculate mean depth within combined mask
        masked_depth_mean = np.nanmean(np.where(combined_mask, depth_img, np.nan))
        # Generate random number of white points
        num_white_points = np.random.randint(num_white_points_range[0], num_white_points_range[1])
        # Generate random coordinates for white points
        random_x = np.random.randint(0, depth_img.shape[1], size=num_white_points)
        random_y = np.random.randint(0, depth_img.shape[0], size=num_white_points)
        # Generate random sizes for white points in the specified range
        random_sizes = np.random.randint(point_size_range[0], point_size_range[1], size=num_white_points)
        # Create masks for all white points at once
        y, x = np.ogrid[:masked_depth_img.shape[0], :masked_depth_img.shape[1]]
        point_masks = ((x[..., None] - random_x) ** 2 + (y[..., None] - random_y) ** 2) <= random_sizes ** 2
        # Update masked depth image with the white points
        masked_depth_img[np.any(point_masks, axis=2)] = np.random.uniform(1.5, 3.0)

        return masked_depth_img, masked_depth_mean

    def render(self):
        imgs = np.zeros((0, self.img_size[0], self.img_size[1]))
        imgs_depth = np.zeros((0, self.img_size[0], self.img_size[1]))
        # imgs_rgb = np.zeros((self.img_size[0], self.img_size[1], 3))
        for camera_name in self.camera_name:
            if self.render_mode == "human":
                self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                return imgs
            elif self.render_mode != "depth_rgb_array":
                img = self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                if self.imshow_cam and self.render_mode == "rgb_array":
                    cv.imshow(camera_name, cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    cv.waitKey(1)
                # Converts the depth array valued from 0-1 to real meters
                elif self.render_mode == "depth_array":
                    img = self.Dcmm.depth_2_meters(img)
                    if self.imshow_cam:
                        depth_norm = np.zeros(img.shape, dtype=np.uint8)
                        cv.convertScaleAbs(img, depth_norm, alpha=(255.0/img.max()))
                        cv.imshow(camera_name+"_depth", depth_norm)
                        cv.waitKey(1)
                    img = np.expand_dims(img, axis=0)
            elif self.render_mode == "rgb_array":
                frame = self.mujoco_renderer.render(self.render_mode, camera_name=self.camera_name[0])
                # 保存帧
                cv.imwrite(f'frames/frame_{self.step_count}.png', cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                self.step_count += 1
                return frame
            else:
                img_rgb = self.mujoco_renderer.render(
                    "rgb_array", camera_name = camera_name
                )
                img_depth = self.mujoco_renderer.render(
                    "depth_array", camera_name = camera_name
                )   
                # Converts the depth array valued from 0-1 to real meters
                img_depth = self.Dcmm.depth_2_meters(img_depth)
                img_depth, _ = self.preprocess_depth_with_mask(img_rgb, img_depth)
                if self.imshow_cam:
                    cv.imshow(camera_name+"_rgb", cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
                    cv.imshow(camera_name+"_depth", img_depth)
                    cv.waitKey(1)
                img_depth = cv.resize(img_depth, (self.img_size[1], self.img_size[0]))
                img_depth = np.expand_dims(img_depth, axis=0)
                imgs_depth = np.concatenate((imgs_depth, img_depth), axis=0)
            # Sync the viewer (if exists) with the data
            if self.Dcmm.viewer != None: 
                self.Dcmm.viewer.sync()
                # 测试/调试时给 viewer 留一点显示时间，否则肉眼会感觉画面“飞过去了”
                if self.viewer_sleep > 0:
                    time.sleep(self.viewer_sleep)
        if self.render_mode == "depth_rgb_array":
            # Only keep the depth image
            imgs = imgs_depth
        return imgs

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        if self.Dcmm.viewer != None: self.Dcmm.viewer.close()

    def run_test(self):
        global cmd_lin_x, cmd_lin_y, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
        self.reset()
        # tidybot Tracking 的动作维度是 10：
        # 2 维底盘 + 7 维机械臂 + 1 维夹爪占位
        action = np.zeros(10)
        while True:
            # Keyboard control
            action[0:2] = np.array([cmd_lin_x, cmd_lin_y])
            if trigger_delta:
                print("delta_xyz: ", delta_xyz)
                action[2:9] = np.ones(7) * delta_xyz
                trigger_delta = False
            else:
                action[2:9] = np.zeros(7)
            if trigger_delta_hand:
                print("delta_xyz_hand: ", delta_xyz_hand)
                action[9:10] = np.ones(1) * delta_xyz_hand
                trigger_delta_hand = False
            else:
                action[9:10] = np.zeros(1)
            base_tensor = action[:2]
            arm_tensor = action[2:9]
            hand_tensor = action[9:10]
            actions_dict = {
                'arm': arm_tensor,
                'base': base_tensor,
                'hand': hand_tensor
            }
            observation, reward, terminated, truncated, info = self.step(actions_dict)

if __name__ == "__main__":
    os.chdir('../../')
    parser = argparse.ArgumentParser(description="Args for DcmmVecEnv")
    parser.add_argument('--viewer', action='store_true', help="open the mujoco.viewer or not")
    parser.add_argument('--imshow_cam', action='store_true', help="imshow the camera image or not")
    args = parser.parse_args()
    print("args: ", args)
    env = DcmmVecEnv(task='Tracking', object_name='object', render_per_step=False, 
                    print_reward=False, print_info=False, 
                    print_contacts=False, print_ctrl=False, 
                    print_obs=False, camera_name = ["wrist"],
                    render_mode="rgb_array", imshow_cam=args.imshow_cam, 
                    viewer = args.viewer, object_eval=False,
                    env_time = 2.5, steps_per_policy=20)
    env.run_test()
