# catch_it 笔记

## 1. 项目一句话

这是一个基于 MuJoCo 的机器人强化学习项目。
环境负责抛出 `object`、推进物理仿真并计算奖励；策略网络负责根据观测输出动作；PPO 根据采样轨迹不断更新策略，让机器人学会完成任务。

我当前只关注：

- `Tracking`
- 新机器人模型：`tidybot`
- 目标：让夹爪末端接近并碰到 `object`

---

## 2. 当前任务设定

### 当前只做 Tracking

- 不做 `Catching`
- 使用 `tidybot`：移动底盘 + 7 关节机械臂 + 夹爪
- 保留原来的 `object` 抛掷任务
- 目标是让夹爪末端接近并碰到 `object`

### 当前成功判定

Tracking 成功满足任意一个条件即可：

1. `object` 真实接触到夹爪 pad geom
2. `ee_distance < 0.02`

解释：

- `ee_distance`：夹爪末端到物体的距离
- 现在把“足够接近”也算成功，但阈值已经收紧到 `2cm`

---

## 3. 推荐阅读顺序

如果想把项目主线读懂，按这个顺序看：

1. `README.md`
2. `train_DCMM.py`
3. `configs/config.yaml`
4. `configs/train/DcmmPPO.yaml`
5. `configs/env/DcmmCfg.py`
6. `gym_dcmm/envs/DcmmVecEnv.py`
7. `gym_dcmm/agents/MujocoDcmm.py`
8. `gym_dcmm/algs/ppo_dcmm/ppo_dcmm_track.py`
9. `gym_dcmm/algs/ppo_dcmm/models_track.py`
10. `gym_dcmm/algs/ppo_dcmm/experience.py`

---

## 4. 项目主流程

### 入口

训练入口在 `train_DCMM.py`。

主线是：

1. 读取配置
2. 创建环境 `env`
3. 实例化 `PPO_Track`
4. 调 `agent.train()`

### 训练总流程

```text
env.reset()
  ->
得到初始观测 obs
  ->
策略网络根据 obs 产生动作
  ->
env.step(action)
  ->
得到 next_obs / reward / done
  ->
把一段轨迹存进 buffer
  ->
用 PPO 更新 actor-critic 网络
  ->
重复很多轮
```

### PPO_Track 里最重要的函数

- `train()`：总训练循环
- `train_epoch()`：一轮 PPO 训练
- `play_steps()`：与环境交互、采样轨迹
- `obs2tensor()`：把环境观测拼成神经网络输入
- `action2dict()`：把网络动作还原成环境动作格式

---

## 5. 当前 tidybot Tracking 的观测与动作

### Tracking 观测维度

- `obs_dim = 25`

### Tracking 动作维度

- `act_dim = 9`

### 动作组成

- `base(2)`：底盘平面动作
- `arm(7)`：7 个关节增量

### 观测组成

- `base.v_lin_2d`
- `arm.ee_pos3d`
- `arm.ee_quat`
- `arm.ee_v_lin_3d`
- `arm.joint_pos(7维)`
- `object.pos3d`
- `object.v_lin_3d`

---

## 6. 控制链

当前 tidybot Tracking 的控制链是：

```text
PPO 输出动作
  ->
base(2) + arm(7)
  ->
环境把动作转换成目标控制量
  ->
MuJoCo 执行 ctrl
  ->
机器人运动
  ->
环境读取新状态并计算 reward
```

### 和旧版本的区别

旧机器人更依赖 `IK + PID`。现在 tidybot Tracking 先采用更直接的关节空间控制思路：

- 底盘：目标平面位姿/速度控制
- 手臂：7 个关节增量控制
- 夹爪：Tracking 中先不作为主要学习目标

---

## 7. PPO 训练流程

### 一轮训练的两个阶段

PPO 一轮分成两段：

1. 采样阶段
2. 更新阶段

### 采样阶段

```text
set_eval()
  ->
play_steps()
  ->
用当前策略和环境交互 horizon_length 步
  ->
存 obs / action / reward / done / value / logprob
```

### 更新阶段

```text
set_train()
  ->
从 storage 里按 mini-batch 取数据
  ->
forward() 重新评估旧动作
  ->
计算 actor loss / critic loss / entropy / kl
  ->
backward()
  ->
optimizer.step()
```

### act 和 forward 的区别

- `act()`：采样时用，负责真正产生动作
- `forward()`：训练时用，负责重新评估旧动作在当前策略下的概率和值

一句话：

- `act()` 是“产生动作”
- `forward()` 是“评估旧动作”

---

## 8. Actor-Critic 的理解

### Actor

作用：输出动作分布参数。

- `mu`：动作均值
- `sigma`：动作标准差

训练时从高斯分布里采样动作：

\[
a_i \sim \mathcal{N}(\mu_i, \sigma_i)
\]

### Critic

作用：输出当前状态价值：

\[
V(s)
\]

它不是直接出动作，而是估计“当前状态未来大概值多少钱”。

---

## 9. reward、return、episode_reward 的区别

### reward

环境每一步立刻给的奖励。

\[
r_t
\]

### episode_reward

一整局 episode 所有 reward 的总和。

\[
episode\_reward = r_0 + r_1 + r_2 + \cdots + r_T
\]

它主要用于：

- 看训练效果
- 画 reward 曲线

### return

从某一步开始往后的累计回报目标。
每一步都有一个 `return_t`。

它主要用于：

- 训练 critic
- 计算 advantage

### 关系

- `episode_reward`：整局只有一个
- `return`：每一步都有一个
- `return_0` 往往最接近整局总 reward

---

## 10. GAE 的理解

### 为什么要有 GAE

PPO 不直接只看即时 reward，而要估计：

- 这一步动作是不是比预期更好

### 关键公式

先算 TD 误差：

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

再用 GAE 递推 advantage：

\[
A_t = \delta_t + \gamma \lambda A_{t+1}
\]

最后得到 return：

\[
return_t = A_t + V(s_t)
\]

### 一句话记忆

- `delta`：当前一步比预期好多少
- `advantage`：这个动作值不值得鼓励
- `return`：critic 应该去逼近的目标

---

## 11. 当前 reward 设计（tidybot Tracking）

### 当前关键权重

- `r_base_pos = 1.0`
- `r_ee_pos = 10.0`
- `r_precision = 12.0`
- `r_close = 2.0`
- `r_orient = 0.8`
- `r_touch['Tracking'] = 8.0`
- `r_collision = -5.0`
- `r_constraint = 1.0`

### 当前 reward 逻辑

- 底盘/机械臂基座更接近目标：奖励
- 末端持续接近目标：奖励
- 末端非常接近目标：更大奖励
- 当 `ee_distance < 0.15` 时：给近距离额外奖励
- 夹爪朝向和物体速度方向同向时：给较弱姿态奖励
- 真实接触 pad：给成功奖励
- 动作太大：惩罚
- 碰撞：惩罚
- 关节越界或不合理动作：惩罚

### 当前设计思想

Tracking 现在不是只靠接触奖励，也不是只靠“接近刷分”，而是混合使用：

- 基座/末端接近目标
- 靠近目标
- 进入近距离区域
- 真实接触奖励
- 足够近直接判成功

来驱动学习。

---

## 12. 当前 PPO 正式训练参数

### 环境与采样

- `num_envs = 8`
- `horizon_length = 64`
- `batch_size = 512`

### 更新参数

- `minibatch_size = 64`
- `mini_epochs = 4`
- `learning_rate = 1e-4`
- `gamma = 0.99`
- `tau = 0.95`
- `entropy_coef = 0.0`
- `max_agent_steps = 6000000`

### 动作反归一化

- `action_track_denorm = [1.5, 0.015, 0.01]`

含义：

- 底盘速度缩放：`1.5`
- 7 关节增量缩放：`0.015`
- 夹爪占位量缩放：`0.01`

### 当前视觉训练开关

- `use_visual_object_state = True`
- `use_visual_object_velocity = False`
- `camera_name = base`

一句话：

- PPO 现在先吃 `base` 相机估计出来的物体位置
- 暂时不直接吃视觉速度
- 姿态奖励仍可用仿真真值速度做 shaping

---

## 13. 曲线怎么理解

### 1. `metrics/episode_rewards_per_step`

不是“每一步 reward”，而是：

- 横轴：训练步数 `agent_steps`
- 纵轴：最近若干个 episode 的平均总 reward

### 2. `metrics/episode_lengths_per_step`

表示最近 episode 的平均长度。

### 3. `metrics/episode_success_per_step`

这是最重要的，最接近“成功率”。

它表示：

- 最近结束的那些 episode 里
- 成功的比例

例如：

- `0.2` = 20% 成功率
- `0.8` = 80% 成功率

### 当前判断训练效果时的优先级

1. `episode_success_per_step`
2. `episode_rewards_per_step`
3. `episode_lengths_per_step`

---

## 14. 训练时间预估

当前参数下：

- `num_envs = 8`
- `max_agent_steps = 6000000`

训练时间大约：

- **6 ~ 10 小时**

影响时长的主要因素：

- MuJoCo 仿真
- 并行环境数量
- CPU 占用
- 是否同时开很多程序

注意：

- reward 设计项多一两项，通常不会显著影响总训练时间
- 但会显著影响学得快不快、效果好不好

---

## 15. 常见缩写表

| 缩写            | 中文意思            |
| --------------- | ------------------- |
| `obs`         | 观测                |
| `act`         | 动作                |
| `env`         | 环境                |
| `ppo`         | PPO 强化学习算法    |
| `rl`          | 强化学习            |
| `ee`          | 末端执行器          |
| `pos`         | 位置                |
| `quat`        | 四元数              |
| `v`           | 速度                |
| `lin`         | 线性                |
| `2d`          | 二维                |
| `3d`          | 三维                |
| `v_lin_2d`    | 二维线速度          |
| `v_lin_3d`    | 三维线速度          |
| `ee_pos3d`    | 末端三维位置        |
| `ee_quat`     | 末端姿态四元数      |
| `ee_v_lin_3d` | 末端三维线速度      |
| `joint_pos`   | 关节位置/关节角     |
| `ctrl`        | 控制量              |
| `done`        | 回合结束            |
| `terminated`  | 失败结束            |
| `truncated`   | 正常截断结束        |
| `mu`          | 均值                |
| `sigma`       | 标准差              |
| `value`       | 状态价值            |
| `advantage`   | 优势函数            |
| `return`      | 回报                |
| `kl`          | KL 散度             |
| `lr`          | 学习率              |
| `ik`          | 逆运动学            |
| `pid`         | PID 控制            |
| `qpos`        | MuJoCo 中的广义位置 |
| `qvel`        | MuJoCo 中的广义速度 |

---

## 16. 我现在最该关注什么



tracking_object_high_height = np.array([1.05, 1.45])#球出生高度

tracking_object_forward_speed = np.array([1.5, 2.2])  # 主要沿 -Y 飞向机器人

tracking_object_lateral_speed = 0.05# 左右摆动

tracking_object_vertical_speed = np.array([2.3, 2.8])



当前最重要的不是继续加很多新功能，而是先回答这几个问题：

1. tidybot Tracking 能不能稳定训练
2. `episode_success_per_step` 能不能明显高于以前的几个百分点
3. reward 上升时，success 是否也跟着上升
4. 当前“近距离即成功”的设计是否合适

一句话：

先把 **tidybot Tracking 训起来并稳定提高成功率**，再考虑后续更复杂的任务。

tracking_object_high_height = np.array([1.05, 1.45])#球出生高度

tracking_object_forward_speed = np.array([1.5, 2.2])  # 主要沿 -Y 飞向机器人

tracking_object_lateral_speed = 0.05# 左右摆动

tracking_object_vertical_speed = np.array([2.3, 2.8])

---

## 17. 视觉位置训练版记录

### 当前目标

- 用 `base` 相机替代 MuJoCo 真值物体位置
- 把相机看到的物体 3D 信息传给 PPO
- 先做 Tracking，不做 Catching

### 当前视觉链路

```text
base 相机 RGB-D
  ->
红色目标检测
  ->
深度反投影到 3D
  ->
转换到相对底盘坐标
  ->
传给 PPO 的 object.pos3d
```

当前说明：

- 视觉位置已经接通
- 视觉速度也能估，但误差偏大
- 所以当前训练版先只用视觉位置，不直接用视觉速度

### 相机视角实验结论

对 `fovy = 55 / 60 / 65` 做过多 episode 误差对比后，当前结论是：

- `65` 最稳
- `55` 太窄，后半段更容易丢检
- `60` 居中，但整体仍略弱于 `65`

因此当前 `base` 相机保留：

- `fovy = 65`

### 视觉误差结论

当前更可靠的结论是：

- `pos3d` 误差已经到“可以尝试训练”的程度
- `v_lin_3d` 误差仍然偏大

所以当前训练策略是：

1. 先让 PPO 主要依赖视觉位置学会追踪
2. 先不要让误差较大的视觉速度干扰策略
3. 等位置驱动的 Tracking 稳定后，再继续修速度估计

### 当前正式训练命令

```bash
python train_DCMM.py
```

### 当前测试命令

```bash
python train_DCMM.py test=True num_envs=1 viewer=True imshow_cam=True
```

如果想看视觉估计和真值误差对比：

```bash
python train_DCMM.py test=True num_envs=1 viewer=True imshow_cam=True debug_visual_compare=True
```
