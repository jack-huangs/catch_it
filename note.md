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

### 当前正式相机安装方案

当前已经决定保留“侧边相机”方案，不再退回中线方案。

当前 `base` 相机参数：

- 相机位置：`(-0.089, -0.214, 1.600)`
- 安装方式：位于底盘侧后方的立杆上
- 朝向方式：`targetbody`
- 目标点：`(1.2, 0, 1.0)`
- 视场角：`fovy = 65`

这样做的原因：

- 能更完整地看到来球轨迹
- 比中线方案更不容易被机械臂遮挡
- 在当前实验里，位置估计误差更小，有效检测率也更高

### 视觉误差结论

先做了 30 回合对比测试，再对当前侧边方案单独做了 100 回合大样本统计。

30 回合对比结论：

- 侧边相机平均位置误差约 `0.020 m`
- 中线相机平均位置误差约 `0.028 m`
- 侧边相机有效检测率约 `0.912`
- 中线相机有效检测率约 `0.887`

100 回合侧边相机统计：

- 平均位置误差：`0.0204 m`
- 中位位置误差：`0.0208 m`
- 90 分位位置误差：`0.0241 m`
- 平均速度误差：`0.2986 m/s`
- 中位速度误差：`0.1980 m/s`
- 90 分位速度误差：`0.3767 m/s`
- 速度方向平均夹角误差：`3.19°`
- 速度大小平均绝对误差：`0.2257 m/s`
- 有效检测率：`0.9306`

当前更可靠的总结是：

- `pos3d` 已经比较准，平均误差大约 `2 cm`
- `v_lin_3d` 现在也比以前稳定很多
- 速度方向已经比较准，做姿态奖励是可行的
- 但当前正式训练版依然先以“视觉位置”为主，更稳妥

所以当前训练策略是：

1. 先让 PPO 主要依赖视觉位置学会追踪
2. 先不要让误差较大的视觉速度干扰策略
3. 等位置驱动的 Tracking 稳定后，再继续修速度估计

### 当前正式训练命令

```bash
python train_DCMM.py
```

当前默认配置含义：

- 使用 `base` 视觉相机
- 使用视觉 `object.pos3d`
- 暂时不把视觉速度直接喂给 PPO
- `max_agent_steps = 6000000`
- `minibatch_size = 64`
- `mini_epochs = 4`

### 当前测试命令

```bash
python train_DCMM.py test=True num_envs=1 viewer=True imshow_cam=True
```

如果想看视觉估计和真值误差对比：

```bash
python train_DCMM.py test=True num_envs=1 viewer=True imshow_cam=True debug_visual_compare=True
```

如果要测试指定模型：

```bash
python train_DCMM.py test=True num_envs=1 viewer=True imshow_cam=True checkpoint_tracking="outputs/Dcmm_Quick/2026-04-22/17:59:05/nn/best_reward_47.55.pth"
```

### 视觉感知代码小白版讲解

好，我们这次不一下子扎进所有细节，就按你最需要的方式来：

**先回答一句：视觉感知这套代码，到底在干嘛？**

它做的是这条链路：

**base 相机拍 RGB 图和深度图**
→ **在 RGB 图里找到红球**
→ **在深度图里拿到这个红球的距离**
→ **把“图像里的点 + 深度”变成三维坐标**
→ **再根据连续几帧估计速度**
→ **最后把这个位置/速度送给 PPO 当作物体状态**

也就是：

**相机看球**
→ **算出球在哪**
→ **算出球往哪飞**
→ **告诉控制策略**

---

# 1. 先看开关：什么时候走视觉，什么时候不走视觉

先看配置文件：
[DcmmCfg.py](/home/jack/桌面/catch_it/configs/env/DcmmCfg.py)

这里最关键的是这一段：

```python
vision_config = {
    "use_visual_object_state": False,
    "use_visual_object_velocity": False,
    "camera_name": "base",
    "min_depth": 0.1,
    "max_depth": 8.0,
    "fallback_to_ground_truth": False,
}
```

你先把它理解成“视觉总开关设置”。

每个参数意思：

- `use_visual_object_state`
  - `False`：不用相机，直接用 MuJoCo 真值
  - `True`：用相机估计出来的位置
- `use_visual_object_velocity`
  - `False`：不用视觉速度
  - `True`：把视觉估计出来的速度也喂给 PPO
- `camera_name="base"`
  - 用底盘上的 `base` 相机
- `min_depth=0.1`
  - 太近的深度不要
- `max_depth=8.0`
  - 太远的深度不要
- `fallback_to_ground_truth=False`
  - 如果视觉没看到球，不偷偷退回真值，而是直接当“没看到”

你现在可以先记一句最核心的：

**只有 `use_visual_object_state=True`，这套视觉代码才真正参与 PPO。**

---

# 2. 视觉入口在环境里哪里接上的

看这个文件：
[DcmmVecEnv.py](/home/jack/桌面/catch_it/gym_dcmm/envs/DcmmVecEnv.py)

先看初始化部分：
[DcmmVecEnv.py:202](/home/jack/桌面/catch_it/gym_dcmm/envs/DcmmVecEnv.py:202)

```python
self.use_visual_object_state = bool(DcmmCfg.vision_config.get("use_visual_object_state", False))
self.use_visual_object_velocity = bool(DcmmCfg.vision_config.get("use_visual_object_velocity", True))
self.visual_camera_name = DcmmCfg.vision_config.get("camera_name", "base")
self.visual_fallback_to_ground_truth = bool(DcmmCfg.vision_config.get("fallback_to_ground_truth", False))
self.visual_estimator = None
if self.use_visual_object_state:
    self.visual_estimator = VisualStateEstimator(...)
```

这里在做什么？

你可以理解成：

- 环境启动时，先看配置
- 如果你说“我要用视觉”
- 那就创建一个 `VisualStateEstimator`

这个类名字很重要：

**`VisualStateEstimator` = 视觉状态估计器**

它就是整个视觉链路的总指挥。

---

# 3. 真正开始拿相机图像

看这里：
[DcmmVecEnv.py:515](/home/jack/桌面/catch_it/gym_dcmm/envs/DcmmVecEnv.py:515)

```python
def _render_visual_object_camera(self):
    rgb_img = self.mujoco_renderer.render("rgb_array", camera_name=self.visual_camera_name)
    depth_img = self.mujoco_renderer.render("depth_array", camera_name=self.visual_camera_name)
    depth_img = self.Dcmm.depth_2_meters(depth_img)
    return rgb_img, depth_img
```

这是视觉链路的第一步。

它做了 3 件事：

1. 从 `base` 相机渲染一张 RGB 图
2. 再渲染一张 depth 深度图
3. 把 MuJoCo 的深度值换算成“米”

所以这里输出的是：

- `rgb_img`：彩色图
- `depth_img`：深度图，单位已经是米

简单说：

**先把相机看到的画面拿到手。**

---

# 4. 然后环境调用视觉估计器

继续看：
[DcmmVecEnv.py:526](/home/jack/桌面/catch_it/gym_dcmm/envs/DcmmVecEnv.py:526)

```python
def _get_visual_object_state(self):
    self._ensure_visual_estimator()
    rgb_img, depth_img = self._render_visual_object_camera()
    estimate = self.visual_estimator.update(self.Dcmm, rgb_img, depth_img, self.Dcmm.data.time)
```

这几句非常关键。

意思就是：

- 先确保视觉估计器存在
- 拿到 RGB 和 depth
- 把它们送给：

```python
self.visual_estimator.update(...)
```

所以视觉感知的核心计算，都在这个 `update()` 里面。

---

# 5. 视觉总入口：`VisualStateEstimator`

看这个文件：
[visual_state_estimator.py](/home/jack/桌面/catch_it/gym_dcmm/vision/visual_state_estimator.py)

先看初始化：

```python
self.detector = RedObjectDetector()
self.projector = DepthProjector(min_depth=min_depth, max_depth=max_depth)
self.tracker = GravityKalman3D()
```

这 3 个东西分别干嘛：

- `RedObjectDetector`
  - 在 RGB 图里找红球
- `DepthProjector`
  - 从像素和深度恢复三维位置
- `GravityKalman3D`
  - 根据连续时刻的位置估计速度，并让轨迹更平滑

你可以把它理解成 3 个工人：

1. **检测工人**：找球
2. **三维工人**：算球在哪
3. **跟踪工人**：算球怎么飞

---

# 6. 最核心函数：`update()`

看：
[visual_state_estimator.py:31](/home/jack/桌面/catch_it/gym_dcmm/vision/visual_state_estimator.py:31)

```python
def update(self, dcmm, rgb_image, depth_image, timestamp):
    det = self.detector.detect(rgb_image)
    if not det["valid"]:
        return self._fallback(False)

    depth_value = self.projector.masked_depth_stat(depth_image, det["mask"])
    if depth_value is None:
        return self._fallback(False)

    u, v = det["centroid"]
    world_pos = self.projector.pixel_to_world(dcmm, u, v, depth_value, self.camera_name)
    world_pos, world_vel = self.tracker.update(world_pos, timestamp)
```

这段就是整条视觉链路的主流程。

我带你一行一行看。

---

## 第一步：检测红球

```python
det = self.detector.detect(rgb_image)
```

就是把 RGB 图扔给检测器。

检测器返回的 `det` 里有这些信息：

- `valid`
  - 找没找到球
- `bbox`
  - 球的框
- `centroid`
  - 球中心像素坐标
- `mask`
  - 球在图里的像素区域

---

## 第二步：如果没找到，就返回失败

```python
if not det["valid"]:
    return self._fallback(False)
```

意思是：

- RGB 图里都没找到红球
- 那后面就没法算三维位置了
- 直接走 fallback

你可以把 `fallback` 理解成：

**“这一帧视觉估计失败了，那我返回一个兜底结果。”**

---

## 第三步：在深度图里取球的深度

```python
depth_value = self.projector.masked_depth_stat(depth_image, det["mask"])
```

这里很重要。

RGB 图只告诉你：

- 球在图像里哪里

但它不告诉你：

- 球离相机多远

深度图正好补这个信息。

这句代码的意思是：

- 用前面检测到的球的 `mask`
- 在深度图里只取这块区域
- 算这块区域的代表深度值

---

## 第四步：如果深度也失败，就返回失败

```python
if depth_value is None:
    return self._fallback(False)
```

也就是说：

- 找到球了
- 但这片区域深度值无效
- 那还是没法算三维位置

---

## 第五步：取球中心像素坐标

```python
u, v = det["centroid"]
```

这里：

- `u`：图像横坐标
- `v`：图像纵坐标

也就是：

**球在图像上的中心点。**

---

## 第六步：像素 + 深度 -> 世界坐标

```python
world_pos = self.projector.pixel_to_world(dcmm, u, v, depth_value, self.camera_name)
```

这是视觉链路最核心的一步之一。

现在你已经有：

- 球在图里的位置 `(u, v)`
- 球离相机多远 `depth_value`

这就够了，可以恢复出球的三维位置。

恢复出来的是：

- `world_pos`
- 也就是 **世界坐标系里的三维位置**

---

## 第七步：根据连续时刻估计速度

```python
world_pos, world_vel = self.tracker.update(world_pos, timestamp)
```

这里做的是：

- 当前帧算出一个三维位置
- 再结合上一帧、上上帧
- 估计球的三维速度

输出：

- `world_pos`
- `world_vel`

也就是：

- 位置
- 速度

---

# 7. 红球检测器是怎么工作的

看这个文件：
[detector.py](/home/jack/桌面/catch_it/gym_dcmm/vision/detector.py)

看核心函数：
[detector.py:20](/home/jack/桌面/catch_it/gym_dcmm/vision/detector.py:20)

```python
hsv = cv.cvtColor(rgb_image, cv.COLOR_RGB2HSV)
```

先把 RGB 图转成 HSV 图。

为什么？

因为用 HSV 找颜色，通常比直接用 RGB 更稳定。

---

然后它定义了两段红色范围：

```python
lower1 = np.array([0, 80, 40], dtype=np.uint8)
upper1 = np.array([10, 255, 255], dtype=np.uint8)
lower2 = np.array([170, 80, 40], dtype=np.uint8)
upper2 = np.array([180, 255, 255], dtype=np.uint8)
```

这是因为 HSV 里红色分布在两头，所以要分两段取。

---

然后做颜色分割：

```python
mask1 = cv.inRange(hsv, lower1, upper1)
mask2 = cv.inRange(hsv, lower2, upper2)
mask = cv.bitwise_or(mask1, mask2)
```

意思就是：

- 把“红色像素”筛出来
- 得到一张二值图 `mask`

白色是红球，黑色是背景。

---

再做一下形态学处理：

```python
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
```

你可以简单理解成：

- 去小噪声
- 把球区域变得更干净

---

然后找轮廓：

```python
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
```

就是在白色区域里找“一个个连通块”。

如果找到多个，就取面积最大的那个：

```python
contour = max(contours, key=cv.contourArea)
```

意思是：

**默认最大的红色块就是球。**

---

最后算中心点：

```python
moments = cv.moments(contour)
cx = moments["m10"] / moments["m00"]
cy = moments["m01"] / moments["m00"]
```

这就是球在图像里的中心坐标。

所以 `detector.py` 的本质非常简单：

**按颜色把红球抠出来，再算中心点。**

---

# 8. 深度怎么变成三维位置

看：
[depth_projector.py](/home/jack/桌面/catch_it/gym_dcmm/vision/depth_projector.py)

先看这个函数：

```python
def masked_depth_stat(self, depth_image, mask):
    valid = (mask > 0) & np.isfinite(depth_image)
    valid &= (depth_image > self.min_depth) & (depth_image < self.max_depth)
    if not np.any(valid):
        return None
    return float(np.median(depth_image[valid]))
```

这段做的是：

- 在 `mask` 覆盖的区域里取深度值
- 去掉无效深度
- 去掉太近/太远的异常值
- 最后取 **中位数**

为什么用中位数？

因为中位数比平均值更不容易被异常点带偏。

所以这里得到的是：

**球的大致深度距离。**

---

再看这个函数：

```python
def pixel_to_world(self, dcmm, u, v, depth, camera_name):
    _, pos_w = dcmm.pixel_2_world(u, v, depth, camera=camera_name)
    return np.asarray(pos_w, dtype=np.float32)
```

它把真正的计算交给了：

```python
dcmm.pixel_2_world(...)
```

所以我们继续看它。

---

# 9. `pixel_2_world()` 真正怎么把像素变成三维点

看：
[MujocoDcmm.py:482](/home/jack/桌面/catch_it/gym_dcmm/agents/MujocoDcmm.py:482)

这里核心是：

```python
pixel_coord = np.array([pixel_x, pixel_y, 1]) * (depth)
pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
```

这一步的直觉理解是：

- 像素坐标是二维的
- 加上深度后，就能恢复出相机坐标系下的三维点

然后又做了坐标轴变换：

```python
pos_c[1] *= -1
pos_c[1], pos_c[2] = pos_c[2], pos_c[1]
```

因为：

- MuJoCo 相机坐标系
- 世界坐标系

这两个坐标轴定义不一样，所以要转一下。

最后：

```python
pos_w = self.cam_rot_mat @ (pos_c) + self.cam_pos
```

意思是：

- 先把相机坐标系里的点旋转到世界方向
- 再加上相机在世界里的位置
- 得到世界坐标 `pos_w`

你可以把这一步理解成：

**先算出“球相对相机在哪”，再根据“相机自己在世界里哪、朝哪看”，换成“球在世界里哪”。**

---

# 10. 速度为什么还要单独估计

因为单帧图像只能告诉你：

- 球现在在哪

但不能直接告诉你：

- 球往哪飞
- 飞多快

所以要看连续多帧。

这就是：
[tracker3d.py](/home/jack/桌面/catch_it/gym_dcmm/vision/tracker3d.py)

它这个类叫：

**`GravityKalman3D`**

名字可以先别怕，先抓住用途：

**根据连续的三维位置，估计更平滑、更合理的三维速度。**

---

它的状态是：

```python
[x, y, z, vx, vy, vz]
```

意思是：

- 位置 `x y z`
- 速度 `vx vy vz`

---

关键点在这里：

```python
measured_vel = (pos3d - self.last_measured_pos) / dt
```

这就是最朴素的速度公式：

**速度 = 位移 / 时间**

也就是你之前一直问的那个逻辑。

---

但直接差分出来的速度会抖，所以代码没有完全相信它，而是又做了融合：

```python
vel_blend = 0.65
self.x[3:] = (1.0 - vel_blend) * self.x[3:] + vel_blend * measured_vel
```

意思是：

- 一部分信 Kalman 自己估出来的平滑速度
- 一部分信最新差分速度
- 两边折中

这样做的好处：

- 响应不会太慢
- 也不会太抖

---

还有一个很关键的地方：

```python
u[2, 0] = -0.5 * self.gravity * dt * dt
u[5, 0] = -self.gravity * dt
```

这就是把**重力**加进去了。

因为你这个球是抛物线飞来的，不是匀速直线飞，所以如果速度模型里完全不考虑重力，估计会更差。

这也是你后面视觉速度变好的一大原因。

---

# 11. 最后怎么送进 PPO

回到环境里看：
[DcmmVecEnv.py:598](/home/jack/桌面/catch_it/gym_dcmm/envs/DcmmVecEnv.py:598)

```python
if self.use_visual_object_state:
    visual_object_state = self._get_visual_object_state()
    self._update_visual_compare_cache(visual_object_state)
    obj_pos3d = visual_object_state["pos3d"]
    if self.use_visual_object_velocity:
        obj_v_lin_3d = visual_object_state["v_lin_3d"]
```

这里做的事特别重要：

原来 PPO 里物体状态是：

- MuJoCo 真值位置
- MuJoCo 真值速度

现在如果打开视觉模式，就换成：

- `visual_object_state["pos3d"]`
- `visual_object_state["v_lin_3d"]`

也就是说：

**PPO 根本不关心这些值是怎么来的。**
它只关心：

- 这里给我的物体位置是多少
- 这里给我的物体速度是多少

所以你后面才能做：

- 真值训练
- 视觉训练
- 真值模型直接接视觉测试

因为 PPO 的接口没变，只是“喂进去的数据来源变了”。

---

# 12. 你可以把整条视觉代码记成一句人话

这一套代码的本质，就是：

**从 base 相机渲染 RGB 和深度图，在 RGB 图里用颜色分割找到红球，在深度图里取红球区域的中位深度，再结合相机内外参把球中心像素反投影成世界三维坐标，最后用带重力先验的 3D Kalman 跟踪器估计球的三维速度，并把位置和速度作为 PPO 的物体观测输入。**

如果你觉得这句太长，小白版就记这个：

**相机先看到球，再算出球的三维位置和速度，最后把这两个量交给 PPO。**

---

# 13. 对你来说最该重点理解的 5 个文件

如果你准备继续深入，我建议按这个顺序反复看：

1. [DcmmCfg.py](/home/jack/桌面/catch_it/configs/env/DcmmCfg.py)
   先懂开关和参数
2. [DcmmVecEnv.py](/home/jack/桌面/catch_it/gym_dcmm/envs/DcmmVecEnv.py)
   懂视觉结果是怎么接进环境观测的
3. [visual_state_estimator.py](/home/jack/桌面/catch_it/gym_dcmm/vision/visual_state_estimator.py)
   懂整条视觉流程
4. [detector.py](/home/jack/桌面/catch_it/gym_dcmm/vision/detector.py)
   懂球是怎么被找到的
5. [depth_projector.py](/home/jack/桌面/catch_it/gym_dcmm/vision/depth_projector.py) 和 [tracker3d.py](/home/jack/桌面/catch_it/gym_dcmm/vision/tracker3d.py)
   懂三维位置和速度怎么来的

---

如果你愿意，下一步我可以继续按这个顺序，**只盯着 `visual_state_estimator.py` 一行一行带你看**。这个文件最适合作为视觉感知的总入口来学。
