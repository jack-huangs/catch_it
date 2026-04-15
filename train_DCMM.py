from __future__ import annotations

import hydra
import torch
import os
import random
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from gym_dcmm.utils.util import omegaconf_to_dict
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_track import PPO_Track
import gymnasium as gym
import gym_dcmm
import datetime
import pytz
# os.environ['MUJOCO_GL'] = 'egl'
# 注册 Hydra 里的自定义解析器：
# 如果用户没传实验名，就使用默认名字；否则使用传入值。
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_name='config', config_path='configs')

def main(config: DictConfig):
    # 使用 spawn 启动多进程，和 PyTorch / 向量环境的兼容性更好。
    torch.multiprocessing.set_start_method('spawn')
    config.test = config.test
    model_path = None

    # 根据任务类型选择对应的 checkpoint，并把路径转成绝对路径。
    if config.task == 'Tracking' and config.checkpoint_tracking:
        config.checkpoint_tracking = to_absolute_path(config.checkpoint_tracking)
        model_path = config.checkpoint_tracking
    elif (config.task == 'Catching_TwoStage' \
        or config.task == 'Catching_OneStage') \
        and config.checkpoint_catching:
        config.checkpoint_catching = to_absolute_path(config.checkpoint_catching)
        model_path = config.checkpoint_catching

    # 为强化学习部分选择运行设备：device_id >= 0 用 GPU，否则用 CPU。
    config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
    # 设置随机种子，尽量保证实验结果可复现。
    config.seed = random.seed(config.seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    # 创建并行环境。
    # 注意：环境层只区分 Tracking / Catching，TwoStage 和 OneStage 的差异在 PPO agent 里。
    env_name = 'gym_dcmm/DcmmVecWorld-v0'
    task = 'Tracking' if config.task == 'Tracking' else 'Catching'
    print("config.num_envs: ", config.num_envs)
    env = gym.make_vec(env_name, num_envs=int(config.num_envs), 
                    # tidybot 原生带 wrist 相机；Tracking 先统一使用它
                    task=task, camera_name=["wrist"],
                    render_per_step=False, render_mode = "rgb_array",
                    object_name = "object",
                    img_size = config.train.ppo.img_dim,
                    imshow_cam = config.imshow_cam, 
                    viewer = config.viewer,
                    print_obs = False, print_info = False,
                    print_reward = False, print_ctrl = False,
                    print_contacts = False, object_eval = config.object_eval,
                    env_time = 2.5, steps_per_policy = 20)

    # 按“输出名/日期/时间”创建实验目录，模型和日志都会写到这里。
    output_dif = os.path.join('outputs', config.output_name)
    # Get the local date and time
    local_tz = pytz.timezone('Asia/Shanghai')
    current_datetime = datetime.datetime.now().astimezone(local_tz)
    current_datetime_str = current_datetime.strftime("%Y-%m-%d/%H:%M:%S")
    output_dif = os.path.join(output_dif, current_datetime_str)
    os.makedirs(output_dif, exist_ok=True)

    # 根据任务名选择不同的 PPO 训练器。
    PPO = PPO_Track if config.task == 'Tracking' else \
          PPO_Catch_TwoStage if config.task == 'Catching_TwoStage' else \
          PPO_Catch_OneStage
    agent = PPO(env, output_dif, full_config=config)

    cprint('Start Training/Testing the Agent', 'green', attrs=['bold'])
    if config.test:
        # 测试模式：恢复模型参数，然后直接跑评估。
        if model_path:
            print("checkpoint loaded")
            agent.restore_test(model_path)
        print("testing")
        agent.test()
    else:
        # 训练模式：初始化 wandb，必要时恢复训练断点，再进入 PPO 主循环。
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.output_name,
            config=omegaconf_to_dict(config),
            mode=config.wandb_mode
        )

        agent.restore_train(model_path)
        agent.train()

        # 训练结束后关闭 wandb，确保日志正常保存。
        wandb.finish()

if __name__ == '__main__':
    main()
