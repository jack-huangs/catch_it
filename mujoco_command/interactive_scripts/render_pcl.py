import pyrallis
import numpy as np
import cv2
import argparse
from envs.utils.camera_utils import pcl_from_obs
from envs.common_mj_env import MujocoEnvConfig
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/dishwasher_wbc.yaml")
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/open_wbc.yaml")
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube_wbc_specified.yaml")
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube_wbc_distractor.yaml")
    args = parser.parse_args()
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    if env_cfg.wbc:
        from envs.mj_env_wbc import MujocoEnv
    else:
        from envs.mj_env_base_arm import MujocoEnv

    env = MujocoEnv(env_cfg)
    env.seed(0)
    env.reset()

    for i in range(100):
        env.step({'gripper_pos':0})

    obs = env.get_obs()

    cv2.imshow('obs', cv2.cvtColor(obs['base1_image'], cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    merged_points, merged_colors = pcl_from_obs(obs, env.cfg)

    print("xmin, xmax:", np.amin(merged_points[:,0]), np.amax(merged_points[:,0]))
    print("ymin, ymax:", np.amin(merged_points[:,1]), np.amax(merged_points[:,1]))
    print("zmin, zmax:", np.amin(merged_points[:,2]), np.amax(merged_points[:,2]))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(merged_points)
    point_cloud.colors = o3d.utility.Vector3dVector(merged_colors)
    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud Viewer")
