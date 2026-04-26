1. 环境安装
使用 mamba (或 conda) 进行环境管理。
```
mamba env create -f linux_env.yml

mamba activate tidybot2
```

2. 快速上手 (Quick Start)
你只需要引入 RobotAPI，即可在你的脚本中直接唤醒机器人并下发控制指令。

运行示例代码
在项目根目录下创建一个 demo.py，并将你的手势控制算法嵌入其中：
```python
import time
import numpy as np
from rmm_api import RobotAPI

# ==========================================
# 1. 初始化引擎
# ==========================================
robot = RobotAPI()
print(">>> 机器人仿真已启动！")

# ==========================================
# 2. 获取初始绝对状态并锁死姿态
# ==========================================
initial_state = robot.get_state()
target_pos = initial_state['arm_pos'].copy()  # 维护一个期望的目标位置
fixed_quat = initial_state['arm_quat'].copy() # 锁死初始姿态，防止手腕震荡

try:
    while True:
        # ==========================================
        # 3. 在此处接入你的手势识别模型！
        # ==========================================
        # 假设手势模型算出了三维空间中的相对位移偏移量 (单位: 米)
        gesture_dx, gesture_dy, gesture_dz = 0.0, 0.0, 0.0005 
        
        # 累加得到绝对目标位置
        target_pos += np.array([gesture_dx, gesture_dy, gesture_dz])
        
        # ==========================================
        # 4. 下发指令驱动机器人
        # ==========================================
        # 注意：底盘会自动配合机械臂运动 (全身协同 WBC)
        robot.step(
            arm_pos=target_pos, 
            arm_quat=fixed_quat, 
            gripper_pos=0.0 # 0.0 为全开，1.0 为全闭
        )
        
except KeyboardInterrupt:
    print(">>> 停止控制，关闭环境...")
    robot.env.close()
```

3. 核心 API 参考 (rmm_api.py)
## RobotAPI()
初始化机器人黑盒底层，默认加载纯净版无菌实验室 (bare_robot.xml) 并拉起 MuJoCo 渲染窗口。

## robot.get_state()
获取当前机器人的实际物理状态。

返回值 (dict):

base_pose: [x, y, theta] 底盘坐标与朝向。

arm_pos: [x, y, z] 机械臂末端的全局绝对坐标。

arm_quat: [x, y, z, w] 机械臂末端的四元数姿态。

gripper_pos: float 夹爪的开合度 (0.0 开 ~ 1.0 闭)。

## robot.step(arm_pos=None, arm_quat=None, gripper_pos=None, base_pose=None)
向系统下发目标指令。如果某个参数传入 None，机器人将保持当前状态或交由底层算法（如 WBC）自动接管。

参数:

arm_pos (np.ndarray): 期望的机械臂末端目标位置 [x, y, z]。

arm_quat (np.ndarray): 期望的姿态 [x, y, z, w]。

gripper_pos (float): 期望的夹爪开合度 0~1。

base_pose (np.ndarray): 期望的底盘坐标 [x, y, theta]（通常省略，让系统自动做全身协同）。

4. 测试运行
```
PYTHONPATH=.:./gsmini_mujoco python rmm_api.py 
```