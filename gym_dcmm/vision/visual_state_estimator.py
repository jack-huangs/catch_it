import numpy as np

from .detector import RedObjectDetector
from .depth_projector import DepthProjector
from .tracker3d import GravityKalman3D


class VisualStateEstimator:
    """
    视觉状态估计总入口：
    RGB/Depth -> 检测 -> 深度统计 -> 3D坐标 -> Kalman速度估计

    第一版先在仿真里工作，后续接真机时只需要替换 detector 和图像输入来源。
    """

    def __init__(self, camera_name="base", min_depth=0.1, max_depth=8.0):
        self.camera_name = camera_name
        self.detector = RedObjectDetector()#在 RGB 图里找红球
        self.projector = DepthProjector(min_depth=min_depth, max_depth=max_depth)#从像素和深度恢复三维位置
        self.tracker = GravityKalman3D()#根据连续时刻的位置估计速度
        self.last_valid = None

    def reset(self):
        """
        每次 episode reset 时，把视觉跟踪器也一起清空。
        否则上一回合的 Kalman 内部速度会“串到”下一回合。
        """
        self.tracker = GravityKalman3D()
        self.last_valid = None

    def update(self, dcmm, rgb_image, depth_image, timestamp):
        det = self.detector.detect(rgb_image)#1111111在 RGB 图里找红球，得到 bbox球的框、centroid球中心像素坐标、mask球在图里的像素区域 等信息
        if not det["valid"]:
            return self._fallback(False)
        #22222222222用前面检测到的球的 mask，在深度图里只取这块区域
        depth_value = self.projector.masked_depth_stat(depth_image, det["mask"])
        if depth_value is None:
            return self._fallback(False)
        #33333取球中心像素坐标，u：图像横坐标，v：图像纵坐标
        u, v = det["centroid"]
        world_pos = self.projector.pixel_to_world(dcmm, u, v, depth_value, self.camera_name)#根据像素坐标和深度值，恢复球在世界坐标系下的三维位置
        world_pos, world_vel = self.tracker.update(world_pos, timestamp)#GravityKalman3D.update根据连续时刻估计速度

        result = {
            "valid": True,
            "world_pos3d": np.asarray(world_pos, dtype=np.float32),
            "world_v_lin_3d": np.asarray(world_vel, dtype=np.float32),
            "depth": float(depth_value),
            "pixel": np.asarray([u, v], dtype=np.float32),
        }
        self.last_valid = result
        return result

    def _fallback(self, valid_flag):
        if self.last_valid is None:
            return {
                "valid": valid_flag,
                "world_pos3d": np.zeros(3, dtype=np.float32),
                "world_v_lin_3d": np.zeros(3, dtype=np.float32),
                "depth": 0.0,
                "pixel": np.zeros(2, dtype=np.float32),
            }
        result = dict(self.last_valid)
        result["valid"] = valid_flag
        return result
