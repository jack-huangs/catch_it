import copy
import json
import os
import time
from datetime import datetime
from threading import Lock
from typing import Callable

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from learning.algos.base_learning import BaseLearning
from robots.base_robot import BaseRobot
from utils.learning import sample_action_space, set_seed


def train(
    args,
    robot: BaseRobot,
    learning_algo: BaseLearning,
    lock: Lock,
    save_model: bool = True,
    file_name: str = "",
) -> None:
    """
    Trains a reinforcement learning agent.

    This function handles the training loop for a reinforcement learning agent,
    interacting with a robot or environment, and optionally saving models and
    performance metrics.

    Parameters
    ----------
    - args : Any
        Command line arguments or configuration object containing various parameters.
    - robot : BaseRobot
        The robot or environment the agent interacts with.
    - learning_algo : BaseLearning
        The learning algorithm to be used for training.
    - lock : Lock
        A threading lock for managing concurrent access.
    - save_model : bool, optional
        Whether to save the model during training. Defaults to True.
    - file_name : str, optional
        Base name for saving model and evaluation results. Defaults to "".

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the directory for saving data does not exist and cannot be created.
    Exception
        For any unexpected errors during training.
    """
    today = datetime.now()

    eval_func: Callable = eval if not args.render else eval_render

    # exit()
    tensorboard_writer = SummaryWriter(f"data/{learning_algo.name}/runs/")

    if file_name != "":
        file_name = (
            file_name
            + "_"
            + today.strftime("%Y%m%d%H%M%S")
            + f"{today.microsecond // 1000:03d}"
        )
    else:
        file_name = today.strftime("%Y%m%d%H%M%S") + f"{today.microsecond // 1000:03d}"
    if not os.path.exists(f"./data/{learning_algo.name}/" + file_name):
        os.makedirs(f"./data/{learning_algo.name}/" + file_name)

    # save experiments config
    save_args(args, f"./data/{learning_algo.name}/" + file_name + "/config.json")

    restart_state = args.state0

    evaluations = [
        [
            eval_func(
                # eval(
                args=args,
                robot=robot,
                learning_algo=learning_algo,
                lock=lock,
                tensorboard_writer=tensorboard_writer,
            ),
            0,
            0,
        ]
    ]

    o, _ = args.reset_function(args, robot, restart_state=restart_state, lock=lock)
    d = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        args.t = episode_timesteps

        if t < args.start_timesteps:
            a = sample_action_space(learning_algo.a_dim, learning_algo.a_max)
        else:
            a = learning_algo.a_noisy(o)

        o2, r, d, info = args.step_function(
            args,
            a,
            robot,
        )

        # Store data in replay buffer
        learning_algo.replay_buffer.push(o, a, o2, r, d, episode_num)

        o = o2
        episode_reward += r

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            learning_algo.optimize(args.batch_size, tensorboard_writer)

        if d:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"[{np.round(t / args.max_timesteps * 100, 3):6.2f}%] Total T: {(t + 1):4d} Episode Num: {(episode_num + 1):4d} Episode T: {(episode_timesteps):3d} Reward: {episode_reward:4.3f}"
            )
            o, _ = args.reset_function(args, robot, restart_state, lock)
            d = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(
                [
                    eval_func(
                        # eval(
                        args,
                        robot=robot,
                        learning_algo=learning_algo,
                        lock=lock,
                        tensorboard_writer=tensorboard_writer,
                    ),
                    t,
                    episode_num,
                ]
            )
            if save_model:
                if not os.path.exists(f"./data/{learning_algo.name}/{file_name}/model"):
                    os.makedirs(f"./data/{learning_algo.name}/{file_name}/model/")
                if not os.path.exists(
                    f"./data/{learning_algo.name}/{file_name}/replay_buffer"
                ):
                    os.makedirs(
                        f"./data/{learning_algo.name}/{file_name}/replay_buffer"
                    )
                # print(evaluations)

                np.save(
                    f"./data/{learning_algo.name}/{file_name}/performance", evaluations
                )
                with open(
                    f"./data/{learning_algo.name}/{file_name}/performance.json", "w"
                ) as f:
                    json.dump(
                        {
                            "avg_reward": np.array(evaluations)[:, 0].tolist(),
                            "t": np.array(evaluations)[:, 1].tolist(),
                            "episode_num": np.array(evaluations)[:, 2].tolist(),
                        },
                        f,
                        indent=4,
                    )
                # save_args(args, f"./data/{learning_algo.name}/" + file_name + "/config.json")

                # Clear the current figure
                plt.clf()
                plt.plot(np.array(evaluations)[:, 1], np.array(evaluations)[:, 0])
                plt.savefig(f"./data/{learning_algo.name}/{file_name}/performance.png")

                learning_algo.save(f"./data/{learning_algo.name}/{file_name}/model/")
                learning_algo.replay_buffer.save(
                    f"./data/{learning_algo.name}/{file_name}/replay_buffer/"
                )

    print("---------------------------------------------------------------------------")
    print("----------------------- Training Ended ------------------------------------")
    print("---------------------------------------------------------------------------")
    if args.exit_on_done:
        print("Finished training... shutting down...")
        exit()


def eval(
    args,
    robot,
    learning_algo: BaseLearning,
    lock: Lock,
    tensorboard_writer: SummaryWriter,
):
    """
    Evaluates a reinforcement learning agent.

    This function evaluates the performance of a trained reinforcement learning
    agent by running it for a specified number of episodes and calculating the
    average reward.

    Parameters
    ----------
    args : Any
        Command line arguments or configuration object.
    robot : BaseRobot
        The robot or environment the agent interacts with.
    learning_algo : BaseLearning
        The learning algorithm used for evaluation.
    lock : Lock
        A threading lock for managing concurrent access.
    tensorboard_writer : SummaryWriter
        TensorBoard writer for logging evaluation metrics.

    Returns
    -------
    float
        The average reward over the evaluation episodes.

    Raises
    ------
    Exception
        For any unexpected errors during evaluation.
    """
    print("[RL]: started eval...")

    set_seed(args.seed_eval)

    cum_reward = 0.0
    total_timesteps = 0
    for i in range(args.eval_episodes):
        print(f"eval progress {i}/{args.eval_episodes}...", end="\r")
        o, _ = args.reset_function(args, robot, args.state0, lock)
        d = False
        episode_timesteps = 0

        while not d:
            episode_timesteps += 1
            args.t = episode_timesteps
            # with timer():
            a = learning_algo.select_action(o).detach().cpu().numpy()
            # with timer("step time"):
            o2, r, d, info = args.step_function(
                args,
                a,
                robot,
            )

            learning_algo.replay_buffer.push(o, a, o2, r, d, -1)
            # self.replay_buffer.push(state, action, next_state, reward, done_bool, -1)
            # print(f"stepping...{env.robot.arm.get_ee_pose().t}, {len(env.robot._traj)}, {done}")
            o = o2
            cum_reward += r
            total_timesteps += 1
    avg_reward = cum_reward / args.eval_episodes

    tensorboard_writer.add_scalar(
        f"Average Reward over {args.eval_episodes} episodes", avg_reward
    )

    # avg_reward = (avg_reward / total_timesteps) * env._max_episode_steps * eval_episodes
    # print(total_timesteps, env._max_episode_steps*eval_episodes)
    print("---------------------------------------")
    print(f"Evaluation over {args.eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    set_seed(args.seed_train)

    return avg_reward


def eval_render(
    args,
    robot,
    learning_algo: BaseLearning,
    lock: Lock,
    tensorboard_writer: SummaryWriter,
):
    """
    Evaluates a reinforcement learning agent with rendering.

    This function evaluates a trained reinforcement learning agent while
    rendering the environment using MuJoCo's viewer. It calculates the average
    reward over a specified number of episodes and logs the metrics.

    Parameters
    ----------
    args : Any
        Command line arguments or configuration object.
    robot : BaseRobot
        The robot or environment the agent interacts with.
    learning_algo : BaseLearning
        The learning algorithm used for evaluation.
    lock : Lock
        A threading lock for managing concurrent access.
    tensorboard_writer : SummaryWriter
        TensorBoard writer for logging evaluation metrics.

    Returns
    -------
    float
        The average reward over the evaluation episodes.

    Raises
    ------
    Exception
        For any unexpected errors during evaluation with rendering.
    """

    render_done = False

    with mj.viewer.launch_passive(model=robot._model, data=robot._data) as viewer:
        # load initial state
        mj.mj_step(robot._model, robot._data)

        # toggle site frame visualization.
        if robot._args.show_site_frames:
            viewer.opt.frame = mj.mjtFrame.mjFRAME_SITE

        # set the size of the rendered coordinate frames
        robot._model.stat.meansize = robot._args.render_size

        while viewer.is_running() or not render_done:
            step_start = time.time()
            set_seed(args.seed_eval)

            cum_reward = 0.0
            total_timesteps = 0
            for i in range(args.eval_episodes):
                print(f"eval progress {i}/{args.eval_episodes}...", end="\r")
                o, _ = args.reset_function(args, robot, args.state0, lock)
                d = False
                episode_timesteps = 0

                while not d:
                    episode_timesteps += 1
                    args.t = episode_timesteps
                    # with timer():
                    a = learning_algo.select_action(o).detach().cpu().numpy()
                    # with timer("step time"):
                    o2, r, d, info = args.step_function(
                        args,
                        a,
                        robot,
                    )

                    # print(f"step {episode_timesteps}...{r}, {d}")

                    learning_algo.replay_buffer.push(o, a, o2, r, d, -1)
                    # self.replay_buffer.push(state, action, next_state, reward, done_bool, -1)
                    # print(f"stepping...{env.robot.arm.get_ee_pose().t}, {len(env.robot._traj)}, {done}")
                    o = o2
                    cum_reward += r
                    total_timesteps += 1

                    viewer.sync()
                    time_until_next_step = robot._model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
            render_done = True
            avg_reward = cum_reward / args.eval_episodes

            tensorboard_writer.add_scalar(
                f"Average Reward over {args.eval_episodes} episodes", avg_reward
            )
            break


def save_args(args, path: str) -> None:
    """
    Saves command line arguments or configuration to a JSON file.

    This function serializes the arguments or configuration object to a JSON
    file, ensuring that callable objects are represented by their names.

    Parameters
    ----------
    args : Any
        Command line arguments or configuration object.
    path : str
        Path to save the JSON file.

    Returns
    -------
    None

    Raises
    ------
    IOError
        If the file cannot be written.
    Exception
        For any unexpected errors during saving.
    """
    try:
        args_dict = copy.deepcopy(vars(args))

        for key, value in args_dict.items():
            if callable(value):
                args_dict[key] = value.__name__

        with open(path, "w") as json_file:
            json.dump(args_dict, json_file, indent=4)

    except IOError as io_error:
        print(f"Error writing to file {path}: {io_error}")
        raise

    except Exception as e:
        print(f"An error occurred while saving arguments: {e}")
        raise
