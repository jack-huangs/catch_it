````instructions
# Copilot Instructions for Catch It (DCMM)

## 🧠 High-level architecture
- **Entry point**: `train_DCMM.py` uses **Hydra** configs (`configs/config.yaml`, `configs/train/DcmmPPO.yaml`) to build a vectorized gym environment and run PPO.
- **Environment**: `gym_dcmm/envs/DcmmVecEnv.py` wraps MuJoCo simulation (via `gym_dcmm/agents/MujocoDcmm.py`) and exposes a Gymnasium API (obs/action spaces, reward shaping, task stages).
- **RL algorithms**: `gym_dcmm/algs/ppo_dcmm/` contains PPO implementations (track vs catch) plus networks (`models_track.py`, `models_catch.py`) and experience buffering.
- **Configs drive behavior**: Most logic (task switching, object randomization, reward weights, logging, rendering) is controlled via Hydra config files.

## 🔑 Core workflows
### Train
```bash
python3 train_DCMM.py test=False task=Tracking num_envs=16
```
### Test (load checkpoint)
```bash
python3 train_DCMM.py test=True task=Tracking num_envs=1 checkpoint_tracking=assets/models/track.pth
```
### Interactive env viewer
```bash
python3 gym_dcmm/envs/DcmmVecEnv.py --viewer
```

## ⚙️ Key conventions & gotchas
- **Task names** (Hydra `task`): `Tracking`, `Catching_TwoStage`, `Catching_OneStage`. This selects the PPO class and what the env expects.
- **Env dim API**: The env exposes dims via `env.call("obs_t_dim")`, `env.call("act_t_dim")`, `env.call("obs_c_dim")`, `env.call("act_c_dim")`. PPO uses these to size networks and normalize inputs.
- **Action denormalization**: `configs/train/DcmmPPO.yaml` fields `action_track_denorm` / `action_catch_denorm` map normalized policy outputs to actual MuJoCo action ranges.
- **Batching constraint**: `num_envs * horizon_length` must be divisible by `minibatch_size` (see `gym_dcmm/algs/ppo_dcmm/ppo_dcmm_track.py`).
- **Output layout**: `configs/config.yaml` sets `hydra.run.dir: .`, so outputs are written to `outputs/<output_name>/<timestamp>/`.
- **MuJoCo GL issues**: If you see `mujoco.FatalError: gladLoadGL error`, set `MUJOCO_GL=egl` (env var or uncomment lines in `train_DCMM.py` / `gym_dcmm/envs/DcmmVecEnv.py`).

## 🛠️ Where to make common changes
- **Reward / termination / stage switching**: `gym_dcmm/envs/DcmmVecEnv.py` (see `compute_reward`, `step`, and `stage_list`).
- **Task parameters & randomization**: `configs/env/DcmmCfg.py` (includes `distance_thresh`, object URDF/mesh selection, random mass/time parameters).
- **RL hyperparameters**: `configs/train/DcmmPPO.yaml` (LR, entropy coeff, horizon length, minibatch size, schedulers, save frequency).
- **Network architecture**: `gym_dcmm/algs/ppo_dcmm/models_track.py` & `models_catch.py`.

## 🧠 Notes for an AI agent
- The codebase is **config-driven**; changing behavior usually means updating Hydra config values rather than editing code.
- The env uses a **vectorized Gym** (`gym.make_vec`) and applies `steps_per_policy` to decouple sim timestep from policy frequency.
- PPO implementations assume a fixed obs/action layout; changing the env observation structure typically requires updates in both the env and corresponding PPO class.

---

If any part of this overview is unclear (e.g., observation layout, `env.call(...)` pattern, or PPO batching), tell me which area to expand.

````
