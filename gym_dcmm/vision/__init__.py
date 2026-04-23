"""
视觉状态估计模块。

这一层的目标不是直接控制机器人，而是把相机图像转换成 PPO 需要的
物体低维状态：

- world_pos3d
- world_v_lin_3d

后续接真机时，只需要替换 detector / depth 输入来源，而不必重写 PPO。
"""

