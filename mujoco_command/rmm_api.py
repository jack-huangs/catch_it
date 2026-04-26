import time
import numpy as np
import pyrallis

from envs.common_mj_env import MujocoEnvConfig
from envs.mj_env_wbc import MujocoEnv 

class RobotAPI:
    def __init__(self, cfg_path="envs/cfgs/door_wbc.yaml"):
        print("[System] 正在唤醒机器人躯壳...")
        self.cfg = pyrallis.parse(config_class=MujocoEnvConfig, config_path=cfg_path)
        
        # 指定加载 bare_robot.xml
        self.cfg.task = 'bare' 
        
        # 启动环境并显示画面
        self.env = MujocoEnv(self.cfg, render_images=True, show_viewer=True)
        self.env.reset()
        print("[System] 机器人已准备就绪")

    def get_state(self):
        """获取当前机器人的位姿"""
        obs = self.env.get_obs()

        return {
            'arm_pos': obs['arm_pos'],       # 机械臂末端 [x, y, z]
            'arm_quat': obs['arm_quat'],     # 机械臂姿态 [x, y, z, w]
            'gripper_pos': obs['gripper_pos'][0] # 夹爪开合度 0~1
        }

    def step(self, arm_pos=None, arm_quat=None, gripper_pos=None, base_pose=None):
        """接收外部传入的目标坐标，驱动机器人"""

        obs = self.env.get_obs()
        
        action = {
            'base_pose': base_pose if base_pose is not None else obs['base_pose'],
            'arm_pos': arm_pos if arm_pos is not None else obs['arm_pos'],
            'arm_quat': arm_quat if arm_quat is not None else obs['arm_quat'],
            'gripper_pos': np.array([gripper_pos]) if gripper_pos is not None else obs['gripper_pos'],
        }
        
        # WBC 坐标系处理
        if not self.cfg.wbc and arm_pos is not None:
             action['arm_pos'] = self.env.global_to_local_arm_pos(action['arm_pos'], action['base_pose'])

        self.env.step(action)
        time.sleep(0.02) # 控制频率限制 (~50Hz)
