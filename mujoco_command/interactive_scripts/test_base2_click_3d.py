import argparse
import sys
import time

import cv2 as cv
import numpy as np
import yaml

from envs.common_real_env_cfg import RealEnvConfig
from envs.utils.cameras import RealSenseCamera


WINDOW_NAME = "base2_rgb"
clicked_pixel = None


def load_cfg(cfg_path: str) -> RealEnvConfig:
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RealEnvConfig(**raw)


def get_camera_cfg(env_cfg: RealEnvConfig, camera_name: str):
    for camera in env_cfg.cameras:
        if camera.name == camera_name:
            return camera
    raise ValueError(f"Camera '{camera_name}' not found in config")


def pixel_to_robot(depth_image, depth_scale, pixel, K, T):
    u, v = pixel
    if depth_image.ndim == 3:
        depth_raw = float(depth_image[v, u, 0])
    else:
        depth_raw = float(depth_image[v, u])

    depth_m = depth_raw * depth_scale
    if depth_m <= 0.0:
        return depth_raw, depth_m, None

    pixel_h = np.array([u, v, 1.0], dtype=float)
    point_camera = np.linalg.inv(K) @ pixel_h * depth_m
    point_camera_h = np.append(point_camera, 1.0)
    point_robot = T @ point_camera_h
    return depth_raw, depth_m, point_robot[:3]


def on_mouse(event, x, y, flags, param):
    del flags, param
    global clicked_pixel
    if event == cv.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)


def main():
    global clicked_pixel
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_cfg",
        type=str,
        default="envs/cfgs/real_wbc.yaml",
        help="Path to the real robot config yaml",
    )
    parser.add_argument("--camera", type=str, default="base2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env_cfg = load_cfg(args.env_cfg)
    camera_cfg = get_camera_cfg(env_cfg, args.camera)
    K = env_cfg.intrinsics[args.camera]
    T = env_cfg.extrinsics[args.camera]

    print(f"Using camera: {camera_cfg.name} ({camera_cfg.type}) serial={camera_cfg.serial}")
    print("K =")
    print(K)
    print("T_camera_to_robot =")
    print(T)

    camera = RealSenseCamera(
        camera_cfg.serial,
        frame_width=args.width,
        frame_height=args.height,
        fps=args.fps,
        use_depth=True,
    )

    try:
        while camera.get_image() is None or camera.get_depth() is None:
            time.sleep(0.05)

        live_intrinsics = camera.get_intrinsics()
        depth_scale = float(live_intrinsics["depth_scale"])
        print(f"Live depth_scale = {depth_scale}")
        print("Press 'c' to inspect image center, left-click any pixel to inspect it, 'q' to quit.")

        cv.namedWindow(WINDOW_NAME)
        cv.setMouseCallback(WINDOW_NAME, on_mouse)

        while True:
            rgb = camera.get_image()
            depth = camera.get_depth()
            if rgb is None or depth is None:
                time.sleep(0.01)
                continue

            frame = cv.cvtColor(rgb.copy(), cv.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            cv.drawMarker(frame, center, (0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=16, thickness=1)
            cv.putText(
                frame,
                "left click: 3D point | c: center point | q: quit",
                (10, 24),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )

            if clicked_pixel is not None:
                u, v = clicked_pixel
                cv.circle(frame, (u, v), 5, (0, 0, 255), -1)

            cv.imshow(WINDOW_NAME, frame)
            key = cv.waitKey(1) & 0xFF

            query_pixel = None
            if key == ord("q"):
                break
            if key == ord("c"):
                query_pixel = center
            elif clicked_pixel is not None:
                query_pixel = clicked_pixel
                clicked_pixel = None

            if query_pixel is None:
                continue

            u, v = query_pixel
            if not (0 <= u < w and 0 <= v < h):
                print(f"Pixel out of range: {(u, v)}")
                continue

            depth_raw, depth_m, point_robot = pixel_to_robot(depth, depth_scale, query_pixel, K, T)
            print(f"\npixel={query_pixel}")
            print(f"depth_raw={depth_raw:.3f}")
            print(f"depth_m={depth_m:.6f}")
            if point_robot is None:
                print("point_robot=invalid depth")
            else:
                print(
                    "point_robot_m="
                    f"[{point_robot[0]:.4f}, {point_robot[1]:.4f}, {point_robot[2]:.4f}]"
                )

    finally:
        camera.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
