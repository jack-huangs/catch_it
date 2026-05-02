import os, sys
sys.path.append(os.path.abspath('../gym_dcmm'))
import math
import time
import torch
import torch.distributed as dist
import cv2

import wandb

import numpy as np
#缓冲区
from .experience import ExperienceBuffer
# from .models import ActorCritic
from .models_track import ActorCritic
from .utils import AverageScalarMeter, RunningMeanStd

from tensorboardX import SummaryWriter

class PPO_Track(object):
    # 初始化 PPO 训练类
    # - env: vectorized gym environment
    # - output_dif: 输出目录（保存模型/日志）
    # - full_config: Hydra 配置（包含训练超参）
    def __init__(self, env, output_dif, full_config):
        self.rank = -1
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- 环境与维度信息 ----
        self.env = env
        self.num_actors = int(self.ppo_config['num_actors'])
        print("num_actors: ", self.num_actors)
        # Tracking 任务只输出底盘 + 机械臂动作，所以这里取 tracking 动作维度
        self.actions_num = self.env.call("act_t_dim")[0]
        print("actions_num: ", self.actions_num)
        self.actions_low = self.env.call("actions_low")[0]#动作空间的下界
        self.actions_high = self.env.call("actions_high")[0]#动作空间的上界
        # Tracking 任务对应的观测维度（不包含手部观测）
        self.obs_shape = (self.env.call("obs_t_dim")[0],)
        # 环境完整动作维度。Tracking 的动作后面会补零对齐成完整动作
        self.full_action_dim = self.env.call("act_c_dim")[0]
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'separate_value_mlp': self.network_config.get('separate_value_mlp', True),
        }
        print("net_config: ", net_config)
        # ActorCritic 同时包含：
        # - actor：根据观测输出动作分布
        # - critic：根据观测输出状态价值 value
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        # 观测标准化器：让输入分布更稳定，更利于网络训练
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        # 价值标准化器：让 critic 预测的 value / return 数值尺度更稳定
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        self.test_video_dir = os.path.join(self.output_dir, 'test_videos')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.init_lr = float(self.ppo_config['learning_rate'])
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.init_lr, eps=1e-5)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.action_track_denorm = self.ppo_config['action_track_denorm']
        self.action_catch_denorm = self.ppo_config['action_catch_denorm']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.reward_scale_value = self.ppo_config['reward_scale_value']
        self.clip_value_loss = self.ppo_config['clip_value_loss']
        # ---- PPO Collect Param ----
        # 每个并行环境一次采样多少步
        self.horizon_length = self.ppo_config['horizon_length']
        # 一轮总样本数 = 每个环境的步数 * 并行环境个数
        self.batch_size = self.horizon_length * self.num_actors
        # 每次做一次梯度下降时，从总样本中取多少条
        self.minibatch_size = self.ppo_config['minibatch_size']
        # 同一批采样数据会被重复训练多少轮
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        # 为了能把一整批数据均匀切成多个 mini-batch，要求能整除
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----#
        # 学习率调度器：控制训练过程中学习率如何变化
        self.lr_schedule = self.ppo_config['lr_schedule']
        if self.lr_schedule == 'kl':
            self.kl_threshold = self.ppo_config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(
                self.init_lr,
                self.ppo_config['max_agent_steps'])
        # ---- Snapshot 
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.episode_success = AverageScalarMeter(200)

        self.obs = None
        self.epoch_num = 0
        # print("self.obs_shape[0]: ", type(self.obs_shape[0]))
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size,
            self.obs_shape[0], self.actions_num, self.device,
        )

        # 用于统计每个 episode 的累计奖励、长度和是否结束
        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)#当前 episode 的累计奖励
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)#当前 episode 的长度
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device) # 1 表示 episode 已经结束，0 表示还在进行中。初始化成全 1，表示一开始就要 reset 环境。
        self.agent_steps = 0 
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.max_test_steps = int(self.ppo_config.get('max_test_steps', 10000))
        self.best_rewards = -10000
        self.save_test_videos = bool(full_config.get('save_test_videos', False))
        self.test_video_episodes = int(full_config.get('test_video_episodes', 8))
        self.test_video_fps = int(full_config.get('test_video_fps', 20))
        self.debug_visual_compare = bool(full_config.get('debug_visual_compare', False))
        self.debug_visual_compare_interval = int(full_config.get('debug_visual_compare_interval', 10))
        self.saved_test_videos = 0
        self.test_video_frames = []
        self.test_episode_index = 0
        self.test_debug_step = 0
        self.next_visual_debug_step = 0
        self.test_total_episodes = 0
        self.test_total_reward_sum = 0.0
        self.test_total_length_sum = 0.0
        self.test_total_success_sum = 0.0
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    # 将当前训练统计数据写入 TensorBoard / wandb
    # - a_losses/c_losses/b_losses: PPO actor/critic/bound losses
    # - entropies: 策略熵，用于鼓励探索
    # - kls: KL 距离，用于监测策略变化
    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        log_dict = {
            'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,
            'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,
            'losses/actor_loss': torch.mean(torch.stack(a_losses)).item(),
            'losses/bounds_loss': torch.mean(torch.stack(b_losses)).item(),
            'losses/critic_loss': torch.mean(torch.stack(c_losses)).item(),
            'losses/entropy': torch.mean(torch.stack(entropies)).item(),
            'info/last_lr': self.last_lr,
            'info/e_clip': self.e_clip,
            'info/kl': torch.mean(torch.stack(kls)).item(),
        }
        for k, v in self.extra_info.items():
            log_dict[f'{k}'] = v

        # log to wandb
        wandb.log(log_dict, step=self.agent_steps)

        # log to tensorboard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    # 切换模型到评估模式（在收集数据时使用）
    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    # 切换模型到训练模式（在优化网络时使用）
    def set_train(self):
        self.model.train()
        if self.normalize_input:#normalize_input 是否对输入进行标准化
            self.running_mean_std.train()#观测标准化器进入训练模式
        if self.normalize_value:
            self.value_mean_std.train()#value / return标准化器进入训练模式

    # 主训练循环：
    # - 反复调用 train_epoch() 进行 PPO 更新
    # - 保存模型（last/best）
    # - 输出训练进度信息
    def train(self):
        # 开始时间
        start_time = time.time()
        # _t：统计“从训练开始到当前”的整体速度
        _t = time.time()
        # _last_t：统计“上一轮到这一轮之间”的局部速度
        _last_t = time.time()
        # 训练开始前先 reset 一次环境，拿到初始观测
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        # agent_steps 表示“已经处理了多少环境步”
        # 这里先记成一个 batch，方便后面的日志和学习率调度统一按 batch 递增
        self.agent_steps = self.batch_size

        # agent_steps训练到现在，一共步数
        # 只要还没达到最大训练步数，就不断重复：
        # 1. 收集一批数据
        # 2. 用 PPO 更新网络
        # 3. 记录日志
        # 4. 保存模型
        while self.agent_steps < self.max_agent_steps:
            # 进入新一轮 epoch
            self.epoch_num += 1
            # train_epoch() 是训练核心：
            # - 先和环境交互，收集一批轨迹
            # - 再基于这批轨迹做多轮 PPO 优化
            # 返回的是这一轮优化过程中记录的各种 loss / 统计量
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            # 这一轮训练结束后，把已经整理好的训练数据引用清空，释放缓存
            self.storage.data_dict = None

            # 如果使用线性学习率衰减，这里按当前总步数更新学习率     ？
            if self.lr_schedule == 'linear':
                self.last_lr = self.scheduler.update(self.agent_steps)
            
            # all_fps：从训练开始到现在的平均速度
            all_fps = self.agent_steps / (time.time() - _t)
            # last_fps：最近这一轮 batch 的处理速度
            last_fps = (
                self.batch_size ) \
                / (time.time() - _last_t)
            # 更新“上一轮结束时间”，供下一轮计算 last_fps
            _last_t = time.time()
            # 拼一条训练进度日志，方便在终端里看当前训练状态
            info_string = f'Agent Steps: {int(self.agent_steps // 1e3):04}K | FPS: {all_fps:.1f} | ' \
                            f'Last FPS: {last_fps:.1f} | ' \
                            f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                            f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                            f'Current Best: {self.best_rewards:.2f}'
            print(info_string)

            # 把这一轮的 actor loss / critic loss / entropy / KL 等写入日志系统
            self.write_stats(a_losses, c_losses, b_losses, entropies, kls)

            # 从滑动统计器中取出最近若干个 episode 的平均表现
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()
            # 继续把 episode 级别的指标写到 TensorBoard 和 wandb
            self.writer.add_scalar(
                'metrics/episode_rewards_per_step', mean_rewards, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_lengths_per_step', mean_lengths, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_success_per_step', mean_success, self.agent_steps)
            wandb.log({
                'metrics/episode_rewards_per_step': mean_rewards,
                'metrics/episode_lengths_per_step': mean_lengths,
                'metrics/episode_success_per_step': mean_success,
            }, step=self.agent_steps)
            # 生成一个带 epoch / 步数 / 奖励信息的 checkpoint 名字
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                # 定期保存快照模型，便于回溯中间训练状态
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                # 每一轮都额外覆盖保存一个 last.pth，表示当前最新模型
                self.save(os.path.join(self.nn_dir, f'last'))

            # 如果这一轮平均奖励超过历史最优，就保存成 best model
            if mean_rewards > self.best_rewards:
                print(f'save current best reward: {mean_rewards:.2f}')
                # 先删除上一个 best 模型，避免目录里堆太多“历史最佳”
                prev_best_ckpt = os.path.join(self.nn_dir, f'best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                # 更新当前最优奖励，并保存新的 best 模型
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))

        # 达到最大训练步数后，输出整个训练阶段的耗时统计
        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    # 保存当前模型权重到文件
    # - name: 保存文件名前缀（会加 '.pth'） 
    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
            'tracking_mlp': self.model.actor_mlp.state_dict(),
            'tracking_mu': self.model.mu.state_dict(),
            'tracking_sigma': self.model.sigma.data
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    # 恢复训练时使用的模型（用于接着训练）
    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location = self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    # 恢复测试时使用的模型（只加载模型参数，不恢复训练状态）
    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location = self.device)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    # 训练一个 epoch（一次完整的数据采集 + 多次 PPO 优化）
    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        # 采样阶段只前向推理，不做参数更新，现在进入“收集数据”阶段，不是“优化参数”阶段。
        self.set_eval()
        # 111111111. 与环境交互 horizon_length 步，把轨迹数据存到经验池里 TODO：play_steps重要
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        # 22222222切回训练模式，现在不再和环境交互，开始用刚才收集的数据更新网络参数
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], [] #策略熵，新旧策略差异
        # 333333333. 开始训练   同一批数据会被反复训练 mini_epochs_num 轮
        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            # storage 已经把总样本切成多个 mini-batch，这里逐批训练
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs = self.storage[i] #从经验池里取一个 mini-batch

                # 训练前先对标准化观测
                obs = self.running_mean_std(obs)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                }
                #444444444!!!!!!!旧动作和旧观测送进当前网络，用当前的新网络（第二minibatch开始），重新评估“旧动作在当前策略下的概率和值”
                res_dict = self.model(batch_dict)# 走的是ActorCritic里的 forward()，
                action_log_probs = res_dict['prev_neglogp']# 当前策略下旧动作的负对数概率
                values = res_dict['values'] # 当前 critic 输出
                entropy = res_dict['entropy'] # 当前策略的熵，当前策略的随机性
                mu = res_dict['mus'] #当前策略分布参数
                sigma = res_dict['sigmas']

                #55555555555算 PPO loss：
                # actor loss：希望高优势动作概率变大，同时限制新旧策略差异不要太大
                ratio = torch.exp(old_action_log_probs - action_log_probs)#当前新策略相对于旧策略，对同一个动作的概率比值
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss：让 critic 预测的 value 接近 return
                if self.clip_value_loss:#如果开了 clip_value_loss，就也对 value 更新做一个 PPO 风格的裁剪，避免 critic 变动过大
                    value_pred_clipped = value_preds + \
                        (values - value_preds).clamp(-self.e_clip, self.e_clip)
                    value_losses = (values - returns) ** 2
                    value_losses_clipped = (value_pred_clipped - returns) ** 2
                    c_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    c_loss = (values - returns) ** 2
                # bounded loss：防止策略均值 mu 跑到过大的区间外
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0

                #把各项 loss 取平均，因为前面这些量是对一个 mini-batch 每个样本分别算的
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef \
                    + b_loss * self.bounds_loss_coef
                
                # 66666666666 反向传播更新网络参数
                self.optimizer.zero_grad()#先清空旧梯度
                loss.backward(retain_graph=True)#再对当前 loss 做反向传播

                # 梯度裁剪，防止训练不稳定
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()#66666666 真正更新网络参数

                #看当前新策略和旧策略差了多少
                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)
                #记录
                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            #一轮 mini-epoch 结束后，算平均 KL
            av_kls = torch.mean(torch.stack(ep_kls))
            kls.append(av_kls)
            #7777777777  更改学习率
            if self.lr_schedule == 'kl':
                self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            elif self.lr_schedule == 'cos':
                self.last_lr = self.adjust_learning_rate_cos(mini_ep)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls
    
    # 将环境返回的观测（dict）转换为模型输入张量
    def obs2tensor(self, obs):
        # 环境给的是结构化字典，网络需要的是一维向量，所以这里手工拼接
        if self.env.call('task')[0] == 'Catching':
            obs_array = np.concatenate((
                        obs["base"]["v_lin_2d"], 
                        obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],
                        obs["object"]["pos3d"], obs["object"]["v_lin_3d"], 
                        obs["hand"],
                        ), axis=1)
        else:
            obs_array = np.concatenate((
                    obs["base"]["v_lin_2d"], 
                    # 新 tidybot Tracking 改成关节空间控制，joint_pos 也要一起喂给策略
                    obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"], obs["arm"]["joint_pos"],
                    obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
                    # obs["hand"],# TODO: TEST
                    ), axis=1)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
        return obs_tensor

    # 将模型输出动作（向量）反归一化并转换成环境可用的动作 dict
    def action2dict(self, actions):
        actions = actions.cpu().numpy()
        # 网络输出在 [-1, 1] 附近，这里按配置放缩回环境真实动作范围
        if self.env.call('task')[0] == 'Tracking':
            base_tensor = actions[:, :2] * self.action_track_denorm[0]
            # tidybot Tracking 的 arm 动作维度改成 7 个关节增量
            arm_tensor = actions[:, 2:9] * self.action_track_denorm[1]
            hand_tensor = actions[:, 9:] * self.action_track_denorm[2]
        else:
            base_tensor = actions[:, :2] * self.action_catch_denorm[0]
            arm_tensor = actions[:, 2:5] * self.action_catch_denorm[1]
            hand_tensor = actions[:, 5:] * self.action_catch_denorm[2]
        actions_dict = {
            'arm': arm_tensor,
            'base': base_tensor,
            'hand': hand_tensor
        }
        return actions_dict

    # 通过模型生成动作（policy）
    # - inference=False: 返回动作 + 价值估计，用于训练
    # - inference=True: 仅返回动作，用于测试
    def model_act(self, obs_dict, inference=False):
        # 先对观测做标准化，再送给 actor-critic
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
        }
        if not inference:
            res_dict = self.model.act(input_dict)
            # 采样时把 value 反标准化回原始尺度，便于后面算 return
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        else:
            res_dict = {}
            res_dict['actions'] = self.model.act_inference(input_dict)
        return res_dict

    # 在环境中执行 horizon_length 步，收集数据用于 PPO 更新
    # - 采集 obs/actions/rewards/dones 到 storage
    # - 处理动作归一化/裁剪、奖励缩放、终止情况
    def play_steps(self):
        # 连续交互 horizon_length 步，得到一整批轨迹数据
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs) # 11111.给环境，得到动作  res_dict表示模型前向计算后返回的一包结果
            # 保存当前时刻的观测、动作、log_prob、value 等信息
            self.storage.update_data('obses', n, self.obs['obs'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # 动作先裁剪到 [-1, 1]，再补零对齐到环境完整动作维度
            actions = res_dict['actions']
            actions[:,:] = torch.clamp(actions[:,:], -1, 1)
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict) #222222. 与环境交互，，返回新观测、奖励、是否结束等信息

            #3333. 得到的新环境，继续转成 tensor，供下一步决策使用
            self.obs = {'obs': self.obs2tensor(obs)}
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # done = terminate（失败）或 truncate（成功/超时），来源于   self.env.step
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
            # 把结束标记和奖励写入经验池
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_scale_value * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            # 444444.下面是在做按 episode 的rewards，lengths，success累计统计
            self.current_rewards += rewards #这个episode的累计奖励
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)#找到结束的episode
            self.episode_rewards.update(self.current_rewards[done_indices])#把结束的episode的整个reward记录下来
            self.episode_lengths.update(self.current_lengths[done_indices])
            self.episode_success.update(torch.tensor(truncates, dtype=torch.float32, device=self.device)[done_indices])
            assert isinstance(infos, dict), 'Info Should be a Dict'
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)#把已结束环境的累计步数清零
            self.current_lengths = self.current_lengths * not_dones

        # 轨迹采完后，再估计一下最后一个状态的 value，用来算 GAE / return
        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']
        #总步数 = 已有步数 + 这一批采集的步数
        self.agent_steps = (self.agent_steps + self.batch_size) 

        # 55555555.根据整条轨迹计算 returns，来算advantage 
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            # 训练 critic 前，把 values 和 returns 拉到相近尺度
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    # 测试阶段和训练采样类似，但不会往 buffer 里存数据，也不会更新网络
    def play_test_steps(self):
        for _ in range(self.horizon_length):
            if self.save_test_videos:
                self.capture_test_frame()
            res_dict = self.model_act(self.obs, inference=True)
            # 测试时直接用策略均值动作，不做采样探索
            actions = res_dict['actions']
            actions[:,:] = torch.clamp(actions[:,:], -1, 1)
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # Map the dones
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
            # Update dones and rewards after env step
            self.current_rewards += rewards
            self.current_lengths += 1
            if self.save_test_videos and self.num_actors == 1 and bool(dones[0]):
                self.capture_test_frame()
                self.finalize_test_episode_video(
                    success=bool(truncates[0]),
                    episode_reward=float(self.current_rewards[0, 0].item()),
                    episode_length=int(self.current_lengths[0].item()),
                )
            done_indices = self.dones.nonzero(as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                done_rewards = self.current_rewards[done_indices]
                done_lengths = self.current_lengths[done_indices]
                done_success = torch.tensor(truncates, dtype=torch.float32, device=self.device)[done_indices]
                # 测试阶段额外累计“从开始到当前”为止的总体统计值，
                # 最终成功率按全部已完成 episode 计算，不再只看最近窗口。
                self.test_total_episodes += int(done_indices.numel())
                self.test_total_reward_sum += float(done_rewards.sum().item())
                self.test_total_length_sum += float(done_lengths.sum().item())
                self.test_total_success_sum += float(done_success.sum().item())
            assert isinstance(infos, dict), 'Info Should be a Dict'
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
        # 测试阶段不更新网络参数，但仍然累计已经测试过的环境步数，
        # 方便外层 test() 按 max_test_steps 正常结束。
        _ = self.model_act(self.obs)
        self.test_steps = (self.test_steps + self.batch_size)

    def capture_test_frame(self):
        """
        从第 0 个环境抓一帧录像图像。
        当前测试录像主要面向 `num_envs=1` 的可视化评估场景。
        """
        if self.saved_test_videos >= self.test_video_episodes:
            return
        try:
            frame = self.env.call("get_record_frame")[0]
        except Exception:
            return
        if frame is None or np.size(frame) == 0:
            return
        self.test_video_frames.append(np.asarray(frame, dtype=np.uint8).copy())

    def finalize_test_episode_video(self, success, episode_reward, episode_length):
        """
        当前 episode 结束后，把缓存帧编码成视频。
        只保存前若干个 episode，避免测试目录无限膨胀。
        """
        self.test_episode_index += 1
        if self.saved_test_videos >= self.test_video_episodes:
            self.test_video_frames = []
            return
        if len(self.test_video_frames) == 0:
            return
        os.makedirs(self.test_video_dir, exist_ok=True)
        status = "success" if success else "fail"
        video_path = os.path.join(
            self.test_video_dir,
            f"episode_{self.test_episode_index:02d}_{status}_r_{episode_reward:.2f}_l_{episode_length}.mp4",
        )
        height, width = self.test_video_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.test_video_fps,
            (width, height),
        )
        for frame in self.test_video_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        self.saved_test_videos += 1
        print(f"saved test video: {video_path}")
        self.test_video_frames = []

    def print_visual_debug_info(self):
        """
        打印最近一步的视觉估计和 MuJoCo 真值，方便判断：
        - 检测是否丢失
        - 位置误差大不大
        - 速度估计是否发抖
        """
        try:
            info = self.env.call("get_visual_debug_info")[0]
        except Exception:
            return
        if info is None:
            print("[visual_debug] no cached visual compare info")
            return
        vision_pos = np.asarray(info["vision_pos3d"])
        gt_pos = np.asarray(info["gt_pos3d"])
        vision_vel = np.asarray(info["vision_v_lin_3d"])
        gt_vel = np.asarray(info["gt_v_lin_3d"])
        print(
            "[visual_debug] "
            f"t={info['time']:.3f} "
            f"valid={info['valid']} "
            f"pos_err={info['pos_error']:.4f} "
            f"vel_err={info['vel_error']:.4f}"
        )
        print(f"  vision_pos3d: {np.array2string(vision_pos, precision=4)}")
        print(f"  gt_pos3d    : {np.array2string(gt_pos, precision=4)}")
        print(f"  vision_vel  : {np.array2string(vision_vel, precision=4)}")
        print(f"  gt_vel      : {np.array2string(gt_vel, precision=4)}")

    # 测试入口：循环调用 play_test_steps，并输出平均表现
    def test(self):
        self.set_eval()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.test_steps = self.batch_size
        self.saved_test_videos = 0
        self.test_video_frames = []
        self.test_episode_index = 0
        self.test_debug_step = 0
        self.next_visual_debug_step = max(self.debug_visual_compare_interval, 1)
        self.test_total_episodes = 0
        self.test_total_reward_sum = 0.0
        self.test_total_length_sum = 0.0
        self.test_total_success_sum = 0.0
        final_rewards = 0.0
        final_lengths = 0.0
        final_success = 0.0

        if self.save_test_videos and self.num_actors != 1:
            print("test video saving currently records only env-0; recommend running with num_envs=1.")
        if self.debug_visual_compare:
            print(
                f"visual compare debug enabled: "
                f"print every ~{max(self.debug_visual_compare_interval, 1)} test steps "
                f"(current horizon_length={self.horizon_length})"
            )

        while self.test_steps < self.max_test_steps:
            self.play_test_steps()
            self.test_debug_step += self.horizon_length
            if self.debug_visual_compare and self.test_debug_step >= self.next_visual_debug_step:
                self.print_visual_debug_info()
                self.next_visual_debug_step += max(self.debug_visual_compare_interval, 1)
            self.storage.data_dict = None
            if self.test_total_episodes > 0:
                mean_rewards = self.test_total_reward_sum / self.test_total_episodes
                mean_lengths = self.test_total_length_sum / self.test_total_episodes
                mean_success = self.test_total_success_sum / self.test_total_episodes
            else:
                mean_rewards = 0.0
                mean_lengths = 0.0
                mean_success = 0.0
            final_rewards = mean_rewards
            final_lengths = mean_lengths
            final_success = mean_success
            print("## Tested Episodes %d ##" % self.test_total_episodes)
            print("overall_mean_rewards: ", mean_rewards)
            print("overall_mean_lengths: ", mean_lengths)
            print("overall_mean_success: ", mean_success)
        print(f"final_test_rewards: {final_rewards}")
        print(f"final_test_lengths: {final_lengths}")
        print(f"final_test_success: {final_success}")
        print(f"final_test_episodes: {self.test_total_episodes}")
        if self.save_test_videos:
            print(f"saved_test_videos: {self.saved_test_videos}")
            # wandb.log({
            #     'metrics/episode_test_rewards': mean_rewards,
            #     'metrics/episode_test_lengths': mean_lengths,
            # }, step=self.agent_steps)

        # 每次测试结束后，额外输出一段汇总信息，方便直接看最终成功率。
        print("==== Final Test Summary ====")
        print("final_mean_rewards: ", final_rewards)
        print("final_mean_lengths: ", final_lengths)
        print("final_mean_success: ", final_success)
        print("final_success_rate: {:.2f}%".format(final_success * 100.0))

    def adjust_learning_rate_cos(self, epoch):
        lr = self.init_lr * 0.5 * (
            1. + math.cos(
                math.pi * (self.agent_steps + epoch / self.mini_epochs_num) / self.max_agent_steps))
        return lr


# 计算旧策略和新策略之间的 KL 散度，用于监控“这次更新改了多少”
def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()

class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    # 如果策略变化太大，就减小学习率；变化太小，就适当增大学习率
    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr

class LinearScheduler:
    def __init__(self, start_lr, max_steps=1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    # 线性衰减学习率：训练越到后期，学习率越小
    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)
