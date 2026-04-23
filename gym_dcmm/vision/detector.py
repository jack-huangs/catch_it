import cv2 as cv
import numpy as np


class RedObjectDetector:
    """
    第一版仿真检测器：
    直接利用 Tracking 任务里 object 的红色外观做分割。

    这样做的目的不是追求最终检测算法，而是先把：
    RGB -> mask/bbox -> depth -> 3D -> velocity
    这条状态估计链路打通。

    后续接真机时，这个类可以被 YOLO / segmentation 模型替换。
    """

    def __init__(self, min_area=20):
        self.min_area = min_area

    def detect(self, rgb_image):
        """
        输入 RGB 图像，输出：
        - bbox
        - centroid
        - mask
        - valid
        """
        hsv = cv.cvtColor(rgb_image, cv.COLOR_RGB2HSV)
        lower1 = np.array([0, 80, 40], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 80, 40], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv.inRange(hsv, lower1, upper1)
        mask2 = cv.inRange(hsv, lower2, upper2)
        mask = cv.bitwise_or(mask1, mask2)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"valid": False, "bbox": None, "centroid": None, "mask": mask}

        contour = max(contours, key=cv.contourArea)
        area = cv.contourArea(contour)
        if area < self.min_area:
            return {"valid": False, "bbox": None, "centroid": None, "mask": mask}

        x, y, w, h = cv.boundingRect(contour)
        moments = cv.moments(contour)
        if moments["m00"] == 0:
            cx = x + w * 0.5
            cy = y + h * 0.5
        else:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

        return {
            "valid": True,
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "centroid": np.array([cx, cy], dtype=np.float32),
            "mask": mask,
        }

