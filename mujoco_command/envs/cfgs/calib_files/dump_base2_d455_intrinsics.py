#!/usr/bin/env python3
import json
from pathlib import Path

import pyrealsense2 as rs


SERIAL = "247122073666"
WIDTH = 640
HEIGHT = 360
FPS = 30


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(SERIAL)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    try:
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intr = color_profile.get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        payload = {
            "camera_name": "base2",
            "source_device": "Intel RealSense D455",
            "serial": SERIAL,
            "resolution": [intr.width, intr.height],
            "depth_units": "meters_after_depth_scale",
            "depth_scale_from_device": depth_scale,
            "distortion": list(intr.coeffs),
            "intrinsic_matrix": [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
        }

        out_path = Path(__file__).with_name("base2_intrinsics_real.json")
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(out_path)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
