import torch
from torch.utils.data import Dataset


def transform_op(arr):
    """
    把张量从 [时间步, 环境数, ...] 变成 [总样本数, ...]

    例如：
    原来是 [horizon_length, num_envs, obs_dim]
    变换后就是 [horizon_length * num_envs, obs_dim]

    这样后面就可以把一整批样本按 mini-batch 切出来训练。
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(Dataset):
    def __init__(
        self, num_envs, horizon_length, batch_size, minibatch_size, obs_dim, act_dim, device):
        self.device = device
        self.num_envs = num_envs
        # 每个环境这次采样会收集多少步轨迹
        self.transitions_per_env = horizon_length

        self.data_dict = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # storage_dict 是“原始经验池”
        # 维度基本都是 [时间步, 并行环境数, 特征维度]
        self.storage_dict = {
            # 观测
            'obses': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.obs_dim),
                dtype=torch.float32, device=self.device),
            # 奖励
            'rewards': torch.zeros(
                (self.transitions_per_env, self.num_envs, 1),
                dtype=torch.float32, device=self.device),
            # critic 给出的状态价值估计
            'values': torch.zeros(
                (self.transitions_per_env, self.num_envs,  1),
                dtype=torch.float32, device=self.device),
            # 旧策略下动作的负对数概率，PPO 计算 ratio 时要用
            'neglogpacs': torch.zeros(
                (self.transitions_per_env, self.num_envs),
                dtype=torch.float32, device=self.device),
            # 这一时刻是否结束
            'dones': torch.zeros(
                (self.transitions_per_env, self.num_envs),
                dtype=torch.uint8, device=self.device),
            # 实际执行的动作
            'actions': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.act_dim),
                dtype=torch.float32, device=self.device),
            # 动作分布均值 mu
            'mus': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.act_dim),
                dtype=torch.float32, device=self.device),
            # 动作分布标准差 sigma
            'sigmas': torch.zeros(
                (self.transitions_per_env, self.num_envs, self.act_dim),
                dtype=torch.float32, device=self.device),
            # 最终算出来的 return（给 critic 当监督信号）
            'returns': torch.zeros(
                (self.transitions_per_env, self.num_envs,  1),
                dtype=torch.float32, device=self.device),
        }

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        # 一整批样本总共能切成多少个 mini-batch
        self.length = self.batch_size // self.minibatch_size

    def __len__(self):
        # DataLoader / for 循环会用到，表示有多少个 mini-batch
        return self.length
        
    # 根据 idx 取出第 idx 个 mini-batch 的数据，供 PPO 训练使用
    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.data_dict.items():
            if type(v) is dict:
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]
        # 返回 PPO 训练这一小批次需要的核心数据
        return input_dict['values'], input_dict['neglogpacs'], input_dict['advantages'], \
            input_dict['mus'], input_dict['sigmas'], input_dict['returns'], input_dict['actions'], \
            input_dict['obses']

    def update_mu_sigma(self, mu, sigma):
        # PPO 在训练过程中会更新策略分布
        # 这里把当前 mini-batch 的新 mu / sigma 写回去，便于后续计算 KL
        start = self.last_range[0]
        end = self.last_range[1]
        self.data_dict['mus'][start:end] = mu
        self.data_dict['sigmas'][start:end] = sigma

    def update_data(self, name, index, val):
        # 往经验池某个字段里写入第 index 个时间步的数据
        # index 通常对应当前 rollout 的第 n 步
        if type(val) is dict:
            for k, v in val.items():
                self.storage_dict[name][k][index,:] = v
        else:
            self.storage_dict[name][index,:] = val

    def compute_return(self, last_values, gamma, tau):
        # 用 GAE（Generalized Advantage Estimation）从后往前计算 advantage 和 return
        last_gae_lam = 0
        mb_advs = torch.zeros_like(self.storage_dict['rewards'])
        for t in reversed(range(self.transitions_per_env)):
            # 最后一个时间步的 next_value 取 rollout 结束后额外估计的 last_values
            if t == self.transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage_dict['values'][t + 1]
            # 如果这一时刻 done=True，后面就不能再 bootstrap
            next_nonterminal = 1.0 - self.storage_dict['dones'].float()[t]
            next_nonterminal = next_nonterminal.unsqueeze(1)
            # TD 误差 delta = r + gamma * V(s') - V(s)
            delta = self.storage_dict['rewards'][t] + \
                gamma * next_values * next_nonterminal - self.storage_dict['values'][t]
            # GAE 递推公式
            mb_advs[t] = last_gae_lam = delta + gamma * tau * next_nonterminal * last_gae_lam #因为是逆序所以是 last_gae_lam 也就是顺序的下一个时间步的 advantage
            # return = advantage + value
            self.storage_dict['returns'][t, :] = mb_advs[t] + self.storage_dict['values'][t]

    def prepare_training(self):
        # 训练前把原始经验池从 [时间步, 环境数, ...] 整理成 [总样本数, ...]
        self.data_dict = {}
        for k, v in self.storage_dict.items():
            self.data_dict[k] = transform_op(v)
        # advantage = return - value
        advantages = self.data_dict['returns'] - self.data_dict['values']
        # 对 advantage 做标准化，PPO 训练会更稳定
        self.data_dict['advantages'] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)).squeeze(1)
        return self.data_dict
