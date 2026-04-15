import argparse
import importlib
import time

import numpy as np
import mujoco


# 通过导入模块对象来覆写其中的 key_callback；
# 这样 DcmmVecEnv 在 launch_passive(viewer) 时，就会绑定我们这里定义的新按键逻辑。
env_mod = importlib.import_module("gym_dcmm.envs.DcmmVecEnv")
DcmmVecEnv = env_mod.DcmmVecEnv


# ------------------------------
# 遥操作状态
# ------------------------------
base_cmd = np.zeros(2, dtype=np.float32)
arm_delta = np.zeros(7, dtype=np.float32)
hand_delta = np.zeros(1, dtype=np.float32)
reset_requested = False
quit_requested = False


def print_controls():
    print("tidybot 手动遥操作（调试模式）")
    print("按键说明：")
    print("  方向键: 调整底盘平面速度")
    print("  k: 底盘速度清零")
    print("  q/a: joint_1 +/-")
    print("  w/s: joint_2 +/-")
    print("  e/d: joint_3 +/-")
    print("  r/f: joint_4 +/-")
    print("  t/g: joint_5 +/-")
    print("  y/h: joint_6 +/-")
    print("  u/j: joint_7 +/-")
    print("  o/p: 夹爪开/关")
    print("  space: reset")
    print("  esc: 退出")
    print("当前模式：默认隐藏 object，且不根据任务成功/失败自动 reset")


def teleop_key_callback(keycode):
    """
    使用 MuJoCo viewer 的原生 key callback，而不是 keyboard 库。
    这样和 `python DcmmVecEnv.py --viewer` 的工作方式一致，稳定性更好。
    """
    global base_cmd, arm_delta, hand_delta, reset_requested, quit_requested

    base_step = 0.3
    arm_step = 0.08
    hand_step = 8.0

    # 方向键：底盘平面速度
    if keycode == 265:   # up
        base_cmd[1] += base_step
        print(f"base_cmd: {base_cmd}")
    elif keycode == 264: # down
        base_cmd[1] -= base_step
        print(f"base_cmd: {base_cmd}")
    elif keycode == 263: # left
        base_cmd[0] -= base_step
        print(f"base_cmd: {base_cmd}")
    elif keycode == 262: # right
        base_cmd[0] += base_step
        print(f"base_cmd: {base_cmd}")
    elif keycode == 256: # esc
        quit_requested = True
    else:
        try:
            ch = chr(keycode).lower()
        except ValueError:
            return

        if ch == " ":
            reset_requested = True
        elif ch == "k":
            base_cmd[:] = 0.0
            print("base_cmd cleared")
        elif ch == "q":
            arm_delta[0] += arm_step
        elif ch == "a":
            arm_delta[0] -= arm_step
        elif ch == "w":
            arm_delta[1] += arm_step
        elif ch == "s":
            arm_delta[1] -= arm_step
        elif ch == "e":
            arm_delta[2] += arm_step
        elif ch == "d":
            arm_delta[2] -= arm_step
        elif ch == "r":
            arm_delta[3] += arm_step
        elif ch == "f":
            arm_delta[3] -= arm_step
        elif ch == "t":
            arm_delta[4] += arm_step
        elif ch == "g":
            arm_delta[4] -= arm_step
        elif ch == "y":
            arm_delta[5] += arm_step
        elif ch == "h":
            arm_delta[5] -= arm_step
        elif ch == "u":
            arm_delta[6] += arm_step
        elif ch == "j":
            arm_delta[6] -= arm_step
        elif ch == "o":
            hand_delta[0] += hand_step
        elif ch == "p":
            hand_delta[0] -= hand_step


def run_headless_smoke():
    """
    用于无图形环境下的最小测试：
    只验证脚本能构建环境并执行几步，不打开 viewer。
    """
    env = DcmmVecEnv(
        task="Tracking",
        object_name="object",
        render_per_step=False,
        print_reward=False,
        print_info=False,
        print_contacts=False,
        print_ctrl=False,
        print_obs=False,
        camera_name=["wrist"],
        render_mode="rgb_array",
        imshow_cam=False,
        viewer=False,
        object_eval=False,
        env_time=2.5,
        steps_per_policy=20,
    )
    obs, info = env.reset()
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(
            {"base": np.zeros(2), "arm": np.zeros(7), "hand": np.zeros(1)}
        )
    print("headless smoke ok")
    env.close()


def configure_manual_debug_mode(env, hide_object=True):
    """
    把训练环境改成“纯手动调试”状态：
    1. 默认把 object 挪到很远处，不干扰机器人观测和控制
    2. 禁用任务成功/失败导致的自动刷新
    3. 关闭物体抛掷逻辑，让场景保持稳定
    """
    env.terminated = False
    env.step_touch = False
    env.reward_touch = 0
    env.stage = "tracking"

    if hide_object:
        # 把物体长期固定在远处；这样 teleop 时就不会出现“空中球”干扰视线，
        # 也不会因为接近/碰撞 object 而触发 Tracking 的成功判定。
        env.object_pos3d = np.array([50.0, 50.0, -5.0], dtype=np.float64)
        env.object_vel6d = np.zeros(6, dtype=np.float64)
        env.object_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        env.object_static_time = 1e9
        env.object_throw = False
        env.Dcmm.set_throw_pos_vel(
            pose=np.concatenate((env.object_pos3d, env.object_q)),
            velocity=np.zeros(6, dtype=np.float64),
        )
        mujoco.mj_forward(env.Dcmm.model, env.Dcmm.data)


def main():
    global arm_delta, hand_delta, reset_requested, quit_requested, base_cmd

    parser = argparse.ArgumentParser(description="tidybot Tracking teleop")
    parser.add_argument("--headless-test", action="store_true", help="只做无图形最小测试")
    parser.add_argument("--show-object", action="store_true", help="手动调试时保留 object")
    args = parser.parse_args()

    if args.headless_test:
        run_headless_smoke()
        return

    # 把环境里的 key_callback 替换成当前脚本的版本。
    env_mod.env_key_callback = teleop_key_callback

    env = DcmmVecEnv(
        task="Tracking",
        object_name="object",
        render_per_step=False,
        print_reward=False,
        print_info=False,
        print_contacts=False,
        print_ctrl=False,
        print_obs=False,
        camera_name=["wrist"],
        render_mode="rgb_array",
        imshow_cam=False,
        viewer=True,
        object_eval=False,
        env_time=2.5,
        steps_per_policy=20,
    )

    obs, info = env.reset()
    configure_manual_debug_mode(env, hide_object=not args.show_object)
    print_controls()

    try:
        while True:
            if quit_requested:
                break

            if reset_requested:
                obs, info = env.reset()
                configure_manual_debug_mode(env, hide_object=not args.show_object)
                reset_requested = False
                arm_delta[:] = 0.0
                hand_delta[:] = 0.0
                continue

            # 纯手动调试模式下，不让训练任务的成功/失败标志影响下一步操控。
            env.terminated = False
            env.step_touch = False

            action_dict = {
                "base": base_cmd.copy(),
                # 关节和夹爪做“一次性增量控制”：按一次键只生效一小步，然后清零
                "arm": arm_delta.copy(),
                "hand": hand_delta.copy(),
            }

            arm_delta[:] = 0.0
            hand_delta[:] = 0.0

            obs, reward, terminated, truncated, info = env.step(action_dict)
            # 手动调试时不自动 reset；即使发生了训练定义里的 done，也只清掉标志继续操控。
            if terminated or truncated:
                env.terminated = False
                env.step_touch = False

            time.sleep(0.03)
    finally:
        env.close()


if __name__ == "__main__":
    main()
