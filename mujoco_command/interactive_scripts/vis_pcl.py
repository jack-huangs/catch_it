import os
os.environ["XDG_RUNTIME_DIR"] = "/tmp"  # Fix Open3D runtime error

import pyrallis
import argparse
import time
import signal
import sys
import threading
import open3d as o3d
from envs.common_real_env_cfg import RealEnvConfig
from envs.real_env_base_arm import RealEnv
from envs.utils.camera_utils import pcl_from_obs
from constants import POLICY_CONTROL_PERIOD

env = None
cleanup_lock = threading.Lock()
cleanup_done = False

def handle_signal(signum, frame):
    global cleanup_done, env
    with cleanup_lock:
        if cleanup_done:
            print("[Force Exit] Cleanup already started. Forcing exit.")
            sys.exit(1)
        print("\n[Signal] Ctrl+C or Ctrl+\\ received. Cleaning up...")
        cleanup_done = True
        if env is not None:
            try:
                env.close()
                print("Closed env.")
            except Exception as e:
                print(f"[Error] Failed to close env cleanly: {e}")
        sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGQUIT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_cfg", type=str, default="envs/cfgs/real_base_arm.yaml")
    args = parser.parse_args()

    try:
        with open(args.env_cfg, "r") as f:
            env_cfg = pyrallis.load(RealEnvConfig, f)
            print("Using Base+Arm env.")

        env = RealEnv(env_cfg)
        env.reset()

        vis = o3d.visualization.Visualizer()
        vis.create_window("Real-Time Point Cloud", width=960, height=540)
        pcd = None

        print("Streaming point cloud. Press Ctrl+C to stop.")
        while True:
            obs = env.get_obs()
            points, colors = pcl_from_obs(obs, env_cfg)

            if points is None or colors is None or len(points) == 0:
                continue

            if pcd is None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(pcd)
            else:
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            time.sleep(POLICY_CONTROL_PERIOD)

    except Exception as e:
        print(f"[Error] Unhandled exception: {e}")

    finally:
        if not cleanup_done:
            try:
                if env is not None:
                    env.close()
                    print("Closed env.")
            except Exception as e:
                print(f"[Error] Cleanup failed in finally block: {e}")

