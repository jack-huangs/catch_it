import os
import numpy as np
from pathlib import Path

## ------------------------------
## 模型与资源路径
## ------------------------------
# 当前配置文件所在路径
path = os.path.realpath(__file__)
root = str(Path(path).parent)
# assets 总目录；XML、模型权重等都从这里找
ASSET_PATH = os.path.join(root, "../../assets")
# print("ASSET_PATH: ", ASSET_PATH)
# 使用 Leap Hand 的完整机器人 XML
XML_DCMM_LEAP_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"
XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/x1_xarm6_leap_right_unseen_object.xml"
# 单独机械臂模型 XML，主要给 IK 求解器用
XML_ARM_PATH = "urdf/xarm6_right.xml"
## 已训练模型权重保存目录
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")

## Catching 任务里，从 tracking 阶段切到 grasping 阶段的距离阈值
distance_thresh = 0.25

## ------------------------------
## 机器人初始关节角
## ------------------------------
# 机械臂 6 个关节的初始角度
arm_joints = np.array([
   0.0, 0.0, -0.0, 3.07, 2.25, -1.5 
])

# 灵巧手 16 个关节的初始角度
hand_joints = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
])

## ------------------------------
## 奖励函数权重
## 这些值会在环境的 compute_reward() 里被用到
## ------------------------------
reward_weights = {
    "r_base_pos": 0.0,   # 底盘/机械臂基座接近目标的奖励权重
    "r_ee_pos": 10.0,    # 末端接近目标的奖励权重
    "r_precision": 10.0, # 末端非常接近目标时的精细奖励
    "r_orient": 1.0,     # 姿态/朝向对准奖励
    "r_touch": {
        'Tracking': 5,   # Tracking 成功接触奖励更大
        'Catching': 0.1  # Catching 中接触只是开始，后面更看重稳定抓取
    },
    "r_constraint": 1.0, # 约束/越界/IK 失败相关惩罚系数
    "r_stability": 20.0, # 抓取后稳定保持的奖励
    "r_ctrl": {
        'base': 0.2, # 底盘动作惩罚权重
        'arm': 1.0,  # 机械臂动作惩罚权重
        'hand': 0.2, # 手部动作惩罚权重
    },
    "r_collision": -10.0, # 碰撞惩罚
}

## ------------------------------
## MuJoCo 渲染相机配置
## ------------------------------
cam_config = {
    "name": "top", # 相机名字，要和 XML 中的 camera 名一致
    "width": 640,  # 渲染宽度
    "height": 480, # 渲染高度
}

## ------------------------------
## 四轮底盘（双阿克曼 / 平行转向）运动学参数
## IKBase 会用到这些参数
## ------------------------------
RangerMiniV2Params = { 
  'wheel_radius': 0.1,                  # in meter //ranger-mini 0.1
  'steer_track': 0.364,                 # in meter (left & right wheel distance) //ranger-mini 0.364
  'wheel_base': 0.494,                   # in meter (front & rear wheel distance) //ranger-mini 0.494
  'max_linear_speed': 1.5,              # in m/s
  'max_angular_speed': 4.8,             # in rad/s
  'max_speed_cmd': 10.0,                # in rad/s
  'max_steer_angle_ackermann': 0.6981,  # 40 degree
  'max_steer_angle_parallel': 1.570,    # 180 degree
  'max_round_angle': 0.935671,
  'min_turn_radius': 0.47644,
}

## ------------------------------
## 机械臂逆运动学（IK）配置
## ------------------------------
ik_config = {
    "solver_type": "QP",  # IK 求解方式：二次规划
    "ps": 0.001,          # 阻尼/步长相关参数
    "λΣ": 12.5,           # 正则化相关参数
    "ilimit": 100,        # IK 迭代次数上限
    "ee_tol": 1e-4        # 末端位姿允许误差
}

## ------------------------------
## 环境随机化参数
## 训练时会随机扰动这些量，让策略更鲁棒
## ------------------------------
## Drive / Steer / Arm / Hand 控制增益随机范围
k_drive = np.array([0.75, 1.25])
k_steer = np.array([0.75, 1.25])
k_arm = np.array([0.75, 1.25])
k_hand = np.array([0.75, 1.25])

## 物体形状、mesh、尺寸、质量、阻尼等随机化范围
object_shape = ["box", "cylinder", "sphere", "ellipsoid", "capsule"]
object_mesh = ["bottle_mesh", "bread_mesh", "bowl_mesh", "cup_mesh", "winnercup_mesh"]
object_size = {
    "sphere": np.array([[0.035, 0.045]]),
    "capsule": np.array([[0.025, 0.035], [0.025, 0.04]]),
    "cylinder": np.array([[0.025, 0.035], [0.025, 0.035]]),
    "box": np.array([[0.025, 0.035], [0.025, 0.035], [0.025, 0.035]]),
    "ellipsoid": np.array([[0.03, 0.03], [0.045, 0.045], [0.045, 0.045]]),
}
object_mass = np.array([0.035, 0.075])
object_damping = np.array([5e-3, 2e-2])
object_static = np.array([0.5, 0.75]) # 物体在真正被抛出前静止悬停的时间范围

## 观测噪声：模拟传感器误差
k_obs_base = 0.01
k_obs_arm = 0.001
k_obs_object = 0.01
k_obs_hand = 0.01

## 动作噪声：执行动作时额外加入的小扰动
k_act = 0.025

## 动作延迟：模拟控制命令不是“立刻生效”
act_delay = {
    'base': [1,], # 底盘动作延迟步数
    'arm': [1,],  # 机械臂动作延迟步数
    'hand': [1,], # 手部动作延迟步数
}

## ------------------------------
## PID 控制器参数
## PPO 先给高层目标，再由 PID 负责稳定追踪目标
## ------------------------------
## 底盘驱动轮 PID
Kp_drive = 5
Ki_drive = 1e-3
Kd_drive = 1e-1
llim_drive = -200
ulim_drive = 200

## 底盘转向轮 PID
Kp_steer = 50.0
Ki_steer = 2.5
Kd_steer = 7.5
llim_steer = -50
ulim_steer = 50

## 机械臂 6 个关节各自的 PID 参数与控制上下限
Kp_arm = np.array([300.0, 400.0, 400.0, 50.0, 200.0, 20.0])
Ki_arm = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3])
Kd_arm = np.array([40.0, 40.0, 40.0, 5.0, 10.0, 1])
llim_arm = np.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0])
ulim_arm = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0])

## 手部 16 个关节的 PID 参数
Kp_hand = np.array([4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      1e-1, 1e-1, 1e-1, 1e-2,])
Ki_hand = 1e-2
Kd_hand = np.array([3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      1e-2, 1e-2, 2e-2, 1e-3,])
llim_hand = -5.0
ulim_hand = 5.0

# hand_mask：哪些手部关节在控制/映射时是活跃的
hand_mask = np.array([1, 0, 1, 1,
                      1, 0, 1, 1,
                      1, 0, 1, 1,
                      0, 1, 1, 1])
