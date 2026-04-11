import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        # 按照 units 给的隐藏层形状，逐层构建一个多层感知机
        # 例如 units=[256,128] 就会得到： 输入维度 -> 256 -> 128 -> 输出维度
        # Linear(input_size,256) -> ELU -> Linear(256,128) -> ELU    ELU 是一种非线性激活函数。
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers) #把刚才堆起来的层，按顺序拼成一个完整网络。

        #神经网络开始训练前，参数不能乱设，初始化方式会影响训练稳定性。
        # orthogonal init of weights
        # hidden layers scale np.sqrt(2)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        # 前向传播：输入一批特征 x，输出 MLP 处理后的特征
        return self.mlp(x)
    
    #这里用的是 orthogonal initialization（正交初始化），目标对weight（wx+b中的w）。
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
# ActorCritic 模型：
# - actor：输出动作分布参数（mu, sigma）
# - critic：输出当前状态的价值 value
class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        separate_value_mlp = kwargs.pop('separate_value_mlp')#critic 是否用同一套网络/kwargs：keyword arguments关键字参数/pop 是 Python 字典的一个方法，按键挖取出对应的值
        self.separate_value_mlp = separate_value_mlp

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]

        # actor 网络
        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        # 如果 separate_value_mlp=True，critic 使用一套独立的网络
        # 否则 actor 和 critic 用不同一个网络（虽然网络的形状相同）
        if self.separate_value_mlp:
            self.value_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        # critic 网络最后一层 最终输出一个数值，表示当前状态的价值
        self.value = torch.nn.Linear(out_size, 1)

        #acotor最后一层 mu 动作分布的均值，如果动作维度是 6，就是6个动作的均值
        self.mu = torch.nn.Linear(out_size, actions_num)
        # sigma 是标准差，控制探索强度，大探索强，可被优化，网络能自己学“探索应该多大”
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)#requires_grad=True这个参数要参与梯度计算，并在训练中被优化器更新。

        #bias参数（wx+b中的b）的初始化
        for m in self.modules():
            #判断m是不是卷及层，tracking没有用到卷及曾，这段不执行
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):#isinstance(...) 是 Python 里的一个函数，判断一个对象是不是某种类型
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            #这段才执行
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

        # policy output layer with scale 0.01
        # value output layer with scale 1    
        # mu，value，actor和critic是网络的最后一层，对最后一层的参数初始化
        torch.nn.init.orthogonal_(self.mu.weight, gain=0.01)#先用正交初始化，然后gain缩小到0.01倍
        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)
    
    def save_actor(self, actor_mlp_path, actor_head_path):
        """
        只保存 actor 相关参数，便于后续单独复用策略网络。
        """
        torch.save(self.actor_mlp.state_dict(), actor_mlp_path)
        torch.save(self.mu.state_dict(), actor_head_path)

    # 训练采样阶段使用，根据环境采样，返回结果，只做前向推理，不记录梯度
    @torch.no_grad()
    def act(self, obs_dict):
        # 根据当前观测构造正态分布，然后“采样”一个动作
        mu, logstd, value = self._actor_critic(obs_dict)#logstd动作分布标准差的对数形式
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)  #为每个动作维度构造一个正态分布
        selected_action = distr.sample() #采样动作
        result = {
            # 当前动作在该分布下的概率，−logπ(a | s)，PPO 计算 ratio 时要用
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            # critic 输出的状态价值
            'values': value,
            # 实际采样出来的动作
            'actions': selected_action,
            # 动作分布均值和标准差，后面算 KL 会用到
            'mus': mu,
            'sigmas': sigma,
        }
        return result
    
    # 测试阶段使用：
    @torch.no_grad()
    def act_inference(self, obs_dict):
        # 不做随机采样，直接返回均值动作 mu，让行为更稳定
        mu, logstd, value = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        # 输入观测 obs，输出动作均值 mu、对数标准差 logstd、状态价值 value
        obs = obs_dict['obs']

        x = self.actor_mlp(obs)
        mu = self.mu(x)
        # critic 可以选择使用单独的 MLP 特征
        if self.separate_value_mlp:
            x = self.value_mlp(obs)
        value = self.value(x)

        sigma = self.sigma
        # 用 tanh 把动作均值限制到 [-1, 1]，后面环境侧再放缩回真实动作范围
        mu = torch.tanh(mu)
        # mu * 0 + sigma：把一维 sigma 广播成和 mu 同 batch 形状的张量
        return mu, mu * 0 + sigma, value

    #多次利用采样，对一批旧样本多次PPO，重新计算当前策略下的动作概率
    def forward(self, input_dict):
        # 训练阶段使用：
        # 给定“prev_actions”和当前观测，重新计算这些动作在新策略下的概率
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        # 熵 entropy：衡量策略分布有多“分散”，常用来鼓励探索
        entropy = distr.entropy().sum(dim=-1)
        # 旧动作在“当前新策略分布”下的负对数概率
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
        }
        return result
