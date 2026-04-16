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
# 当前项目改成使用 tidybot（移动底盘 + 7 关节机械臂 + 夹爪）
ROBOT_MODEL = "tidybot"
# Tracking 任务使用的新机器人 XML；object 会在代码里按旧任务逻辑自动补进场景
XML_DCMM_LEAP_OBJECT_PATH = "urdf/stanford_tidybot2/tidybot.xml"
XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/stanford_tidybot2/tidybot.xml"
# tidybot 当前不再单独使用旧的 xArm6 IK 模型，这里保留同一路径占位
XML_ARM_PATH = "urdf/stanford_tidybot2/tidybot.xml"
## 已训练模型权重保存目录
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")

## Catching 任务里，从 tracking 阶段切到 grasping 阶段的距离阈值
distance_thresh = 0.25

## Tracking 任务里，夹爪末端距离物体足够近时，直接视为成功
## 当前把阈值进一步收紧到 2cm，避免策略只停留在“靠近但不真正完成”
tracking_success_thresh = 0.02
## 当末端进入较近区域时，额外给一小笔奖励，引导策略更愿意继续逼近目标
## 这里把“近距离区域”放宽到 15cm，给策略更平滑的过渡带
tracking_close_bonus_thresh = 0.15

## ------------------------------
## 机器人初始关节角
## ------------------------------
# 底盘初始位姿 [x, y, yaw]：
# 把 tidybot 初始朝向旋到 +Y 方向，这样它会正面对着从前方飞来的物体。
base_init_pose = np.array([0.0, 0.0, np.pi / 2])

# 机械臂 7 个关节的初始姿态：
# 不再直接沿用 tidybot.xml 里较“收拢”的 home，而是改成更适合 Tracking 的前伸待命位。
# 这组角度会让夹爪末端一开始就位于底盘前上方，减少：
# 1. 训练初期为了抬手/伸手产生的大幅摆动
# 2. 末端离物体太远导致的无效探索
# 3. 机械臂贴近机身时更容易出现的视觉“抖动感”
arm_joints = np.array([
   0.0, 0.50, 3.14159265, -1.80, 0.0, 1.10, 1.57079633
])

# Tracking 不训练抓取，这里只保留 1 维夹爪控制量，默认保持张开
hand_joints = np.array([
    0.0,
])

## ------------------------------
## tidybot 关键 body / site / geom 名称
## 环境里会用这些名字取末端位置、姿态和接触信息
## ------------------------------
base_body_name = "base_link"
arm_base_body_name = "gen3/base_link"
ee_body_name = "base"
ee_site_name = "pinch_site"
pad_geom_names = ["right_pad1", "right_pad2", "left_pad1", "left_pad2"]

## ------------------------------
## 奖励函数权重
## 这些值会在环境的 compute_reward() 里被用到
## ------------------------------
reward_weights = {
    "r_base_pos": 1,   # 底盘/机械臂基座接近目标的奖励权重
    "r_ee_pos": 10.0,    # 末端接近目标的奖励权重：适当下调，减少“只靠接近刷分”
    "r_precision": 12.0, # 末端非常接近目标时的精细奖励：下调后避免 reward 与真实成功脱节
    "r_orient": 0.8,     # 姿态/朝向对准奖励：对 tidybot Tracking 只保留较弱引导
    "r_close": 2.0,      # Tracking 近距离额外奖励：保留但弱化，避免单纯停在近距离区域刷分
    "r_touch": {
        'Tracking': 8, # 提高真实成功/接触奖励，让策略更重视“最终完成动作”
        'Catching': 0.1  # Catching 中接触只是开始，后面更看重稳定抓取
    },
    "r_constraint": 1.0, # 约束/越界/IK 失败相关惩罚系数
    "r_ctrl": {
        'base': 0.2, # 底盘动作惩罚权重
        'arm': 1.0,  # 机械臂动作惩罚权重
        'hand': 0.2, # 手部动作惩罚权重
    },
    "r_collision": -5.0, # 碰撞惩罚
}

## ------------------------------
## MuJoCo 渲染相机配置
## ------------------------------
cam_config = {
    "name": "wrist", # tidybot 原生就有 wrist 相机，先用它做渲染/调试
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
# Tracking 当前希望“球抛出”和“机器人开始响应”同步发生，
# 因此把物体静止悬停时间设为 0，避免出现机器人先空追一段时间的现象。
object_static = np.array([0.0, 0.0]) # 物体在真正被抛出前静止悬停的时间范围

# tidybot Tracking 的“来球走廊”：
# 让物体主要从机器人正前方 (+Y) 飞来，只保留少量左右偏移，
# 这样更符合“机器人面对来球”的训练设定。
tracking_object_x_range = np.array([-0.15, 0.15])#表示球在 左右方向 的出生范围
tracking_object_y_range = np.array([2.2, 2.45])#表示球在 前后方向 的出生范围
tracking_object_low_height = np.array([0.85, 1.05])#球出生高度
tracking_object_high_height = np.array([1.05, 1.45])#球出生高度
tracking_object_forward_speed = np.array([1.5, 2.2])  # 主要沿 -Y 飞向机器人
tracking_object_lateral_speed = 0.05                  # 左右摆动
tracking_object_vertical_speed = np.array([2.3, 2.8])

## 观测噪声：模拟传感器误差
k_obs_base = 0.01
k_obs_arm = 0.001
k_obs_object = 0.01
k_obs_hand = 0.01

## 动作噪声：执行动作时额外加入的小扰动
## tidybot Tracking 当前优先追求“平稳控制”，这里先关闭执行噪声，
## 避免底盘和 7 关节在训练初期被随机扰动带得发抖。
k_act = 0.0

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

## tidybot 机械臂 7 个关节的占位 PID 参数
## 当前 Tracking 改成直接输出关节目标位置，这些值只保留兼容接口
Kp_arm = np.array([120.0, 120.0, 120.0, 80.0, 60.0, 40.0, 40.0])
Ki_arm = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
Kd_arm = np.array([5.0, 5.0, 5.0, 3.0, 2.0, 2.0, 2.0])
llim_arm = np.array([-200.0, -200.0, -200.0, -120.0, -80.0, -60.0, -60.0])
ulim_arm = np.array([200.0, 200.0, 200.0, 120.0, 80.0, 60.0, 60.0])

## 夹爪只有 1 个控制量
Kp_hand = np.array([1.0])
Ki_hand = 1e-2
Kd_hand = np.array([1e-2])
llim_hand = -5.0
ulim_hand = 5.0

# hand_mask：Tracking 里夹爪只保留 1 维占位控制量
hand_mask = np.array([1])
