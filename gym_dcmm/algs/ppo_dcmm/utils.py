import torch
import torch.nn as nn
import numpy as np

# 用来统计“最近一段样本”的平均标量值。
# 训练里常拿它统计：
# - episode reward
# - episode length
# - episode success
class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size  # 滑动窗口大小
        self.current_size = 0           # 当前已经统计了多少个样本
        self.mean = 0                   # 当前窗口内的平均值

    def update(self, values):
        # values 一般是一个张量，里面可能放着这一步刚结束的若干个 episode 的统计值
        size = values.size()[0]
        if size == 0:
            return
        # 先算这批新数据自己的平均值
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        # 最多只保留 window_size 个样本的统计效果
        size = np.clip(size, 0, self.window_size)
        # old_size 表示旧窗口里还能保留多少“历史样本权重”
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        # 用“旧均值 + 新均值”做加权平均，得到更新后的窗口均值
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean

# 在线标准化模块：
# 运行时不断统计数据的均值/方差，并把输入归一化到更稳定的尺度。
# 这个项目里主要拿它做两件事：
# 1. 标准化观测 obs
# 2. 标准化 critic 的 value / return
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon  # 防止分母出现 0

        self.norm_only = norm_only   # True: 只除标准差；False: 做完整 (x-mean)/std
        self.per_channel = per_channel  # 是否按通道分别统计均值/方差
        if per_channel:
            # 如果是图像/多通道输入，这里决定“沿哪些维度”求均值方差
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            # 普通向量输入时，只沿 batch 维统计
            self.axis = [0]
            in_size = insize

        # register_buffer 表示：
        # 这些量会跟着模型一起保存/加载，但不是需要梯度更新的参数
        self.register_buffer('running_mean', torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer('running_var', torch.ones(in_size, dtype = torch.float64))
        self.register_buffer('count', torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count):
        # 用“已有统计量 + 当前 batch 统计量”在线更新整体均值/方差
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            # train 模式下，一边用当前 batch 做标准化，一边更新运行中的均值方差
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = \
                self._update_mean_var_count_from_moments(
                    self.running_mean, self.running_var, self.count, mean, var, input.size()[0])

        # 把 running_mean / running_var reshape 成和输入兼容的形状
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        # unnorm=True 时做“反标准化”，通常用于把网络里的 value 恢复到原始尺度
        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                # 只按标准差缩放，不减均值
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                # 标准化公式：(x - mean) / std
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                # 截断极端值，避免异常大输入影响训练稳定性
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y

# 简单自测：验证前 12 维单独标准化时，结果和 17 维输入切片后保持一致
def test_running_mean_std():
    input_data = torch.randn(10, 17)  # 10个样本，每个样本17维
    running_mean_std = RunningMeanStd(insize=(17,))
    running_mean_std_12 = RunningMeanStd(insize=(12,))
    output = running_mean_std(input_data)
    input_data_12 = input_data[:, :12]
    output_12 = running_mean_std_12(input_data_12)

    assert torch.allclose(output[:, :12], output_12), "Output for 12 dimensions does not remain the same."

if __name__ == "__main__":
    test_running_mean_std()
    print("Test passed successfully!")
