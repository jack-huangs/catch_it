import os

import cv2
import numpy as np


class TactileContourRecognizer:
    def __init__(
        self,
        alpha=0.2,
        rows=7,
        cols=9,
        marker_radius=4,
        remap_scale=120000.0,
        depth_scale=5000.0,
        noise_threshold=0.001,
        min_arrow_length=3.0,
        min_contour_area=20.0,
        bg_image_path=None,
    ):
        self.alpha = alpha
        self.rows = rows
        self.cols = cols
        self.marker_radius = marker_radius
        self.remap_scale = remap_scale
        self.depth_scale = depth_scale
        self.noise_threshold = noise_threshold
        self.min_arrow_length = min_arrow_length
        self.min_contour_area = min_contour_area
        self.bg_image_path = bg_image_path

        self._prev_depth = None
        self._marker_base_img = None
        self._marker_pts = []

    def _create_marker_bg(self, width, height):
        if self.bg_image_path and os.path.exists(self.bg_image_path):
            bg = cv2.imread(self.bg_image_path)
            if bg is None:
                bg = np.ones((height, width, 3), dtype=np.uint8) * 255
            else:
                bg = cv2.resize(bg, (width, height))
        else:
            bg = np.ones((height, width, 3), dtype=np.uint8) * 255

        xs = np.linspace(0, width, self.cols + 2)[1:-1].astype(int)
        ys = np.linspace(0, height, self.rows + 2)[1:-1].astype(int)

        marker_points = []
        for x in xs:
            for y in ys:
                cv2.circle(bg, (x, y), self.marker_radius, (0, 0, 0), -1)
                marker_points.append((x, y))

        return bg, marker_points

    def process(self, depth_img):
        if depth_img is None:
            return None

        h, w = depth_img.shape[:2]
        if self._marker_base_img is None or self._marker_base_img.shape[:2] != (h, w):
            self._marker_base_img, self._marker_pts = self._create_marker_bg(w, h)

        depth_f32 = depth_img.astype(np.float32)
        if self._prev_depth is None or self._prev_depth.shape != depth_f32.shape:
            self._prev_depth = depth_f32
        else:
            self._prev_depth = self.alpha * depth_f32 + (1.0 - self.alpha) * self._prev_depth

        depth_blur = cv2.GaussianBlur(self._prev_depth, (21, 21), 10)
        gy, gx = np.gradient(depth_blur)

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x - gx * self.remap_scale).astype(np.float32)
        map_y = (y - gy * self.remap_scale).astype(np.float32)
        distorted = cv2.remap(self._marker_base_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        bg_depth = float(np.max(self._prev_depth))
        depth_diff = np.abs(bg_depth - self._prev_depth)
        contact_mask = (depth_diff >= self.noise_threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(contact_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

        contour_overlay = cv2.cvtColor(contact_mask, cv2.COLOR_GRAY2BGR)
        if contours:
            cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 1)

        for mx, my in self._marker_pts:
            curr_x, curr_y = float(mx), float(my)
            for _ in range(3):
                idx_x = int(np.clip(curr_x, 0, w - 1))
                idx_y = int(np.clip(curr_y, 0, h - 1))
                curr_x = mx + gx[idx_y, idx_x] * self.remap_scale
                curr_y = my + gy[idx_y, idx_x] * self.remap_scale

            start_x = int(np.clip(curr_x, 0, w - 1))
            start_y = int(np.clip(curr_y, 0, h - 1))

            local_diff = depth_diff[start_y, start_x]
            if local_diff < self.noise_threshold:
                continue

            dir_x = curr_x - mx
            dir_y = curr_y - my
            arrow_length = local_diff * self.depth_scale

            if abs(dir_x) < 0.1 and abs(dir_y) < 0.1:
                if arrow_length > 5.0:
                    dir_y = -1.0
                else:
                    continue

            norm = np.sqrt(dir_x * dir_x + dir_y * dir_y) + 1e-6
            arrow_dx = dir_x / norm * arrow_length
            arrow_dy = dir_y / norm * arrow_length

            if arrow_length > self.min_arrow_length:
                end_x = int(np.clip(start_x + arrow_dx, 0, w - 1))
                end_y = int(np.clip(start_y + arrow_dy, 0, h - 1))
                cv2.arrowedLine(
                    distorted,
                    (start_x, start_y),
                    (end_x, end_y),
                    (255, 0, 0),
                    thickness=2,
                    tipLength=0.3,
                )

        return {
            "distorted_markers": distorted,
            "contact_mask": contact_mask,
            "contour_overlay": contour_overlay,
            "contours": contours,
            "depth_smooth": self._prev_depth,
        }
