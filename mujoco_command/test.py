from rmm_api import RobotAPI
import numpy as np
import time

robot = RobotAPI()

initial_state = robot.get_state() #这个调用了api的获取初始位姿
target_pos = initial_state['arm_pos'].copy() #定义了参数
fixed_quat = initial_state['arm_quat'].copy() 

print("开始平滑抬升...")

while True:
    target_pos += np.array([0.0, 0.0, 0.0005]) #机械臂以z的速度上升
    
    robot.step(
        arm_pos=target_pos,
        arm_quat=fixed_quat
    )
    
    time.sleep(0.02)
