import time

import numpy as np

from gym_dcmm.envs.real_robot_bridge import RealRobotBridgeEnv


def main():
    env = RealRobotBridgeEnv()
    try:
        obs, _ = env.reset()
        print("Initial obs keys:", obs.keys())
        print("Initial arm ee_pos3d:", obs["arm"]["ee_pos3d"])
        print("Initial arm joint_pos:", obs["arm"]["joint_pos"])

        zero_action = {
            "base": np.zeros(2, dtype=np.float32),
            "arm": np.zeros(7, dtype=np.float32),
            "hand": np.zeros(1, dtype=np.float32),
        }

        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            print(
                f"step={i} ee_pos3d={obs['arm']['ee_pos3d']} "
                f"base_vel={obs['base']['v_lin_2d']} reward={reward} "
                f"terminated={terminated} truncated={truncated} info={info}"
            )
            time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
