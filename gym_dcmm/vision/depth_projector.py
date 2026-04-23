import numpy as np


class DepthProjector:
    """
    根据 mask / 像素中心和深度值恢复物体 3D 坐标。
    """

    def __init__(self, min_depth=0.1, max_depth=8.0):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def masked_depth_stat(self, depth_image, mask):
        valid = (mask > 0) & np.isfinite(depth_image)
        valid &= (depth_image > self.min_depth) & (depth_image < self.max_depth)
        if not np.any(valid):
            return None
        return float(np.median(depth_image[valid]))

    def pixel_to_world(self, dcmm, u, v, depth, camera_name):
        _, pos_w = dcmm.pixel_2_world(u, v, depth, camera=camera_name)
        return np.asarray(pos_w, dtype=np.float32)

