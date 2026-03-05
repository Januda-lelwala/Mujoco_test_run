import sys
import os
import subprocess

# On macOS, launch_passive requires mjpython. Re-launch under it if needed.
if sys.platform == "darwin" and os.environ.get("_MJPYTHON_RELAUNCHED") != "1":
    mjpython = os.path.join(os.path.dirname(sys.executable), "mjpython")
    env = os.environ.copy()
    env["_MJPYTHON_RELAUNCHED"] = "1"
    sys.exit(subprocess.call([mjpython, __file__] + sys.argv[1:], env=env))

import numpy as np
import mujoco
import mujoco.viewer
from rl import PPO
import torch

# ── Model ────────────────────────────────────────────────────────────────────
# Unitree A1 from mujoco_menagerie (already cached locally)
MJCF_PATH = os.path.expanduser(
    "~/.cache/robot_descriptions/mujoco_menagerie/unitree_a1/scene.xml"
)
m = mujoco.MjModel.from_xml_path(MJCF_PATH)
d = mujoco.MjData(m)

# A1 stands at ~0.30 m body height; alive band is generous to allow movement
A1_ALIVE_MIN = 0.20   # fallen / rolled over
A1_ALIVE_MAX = 0.50   # unreachable in normal operation

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")
print(f"State dim: {m.nq + m.nv}  |  Action dim: {m.nu}")

state_dim = m.nq + m.nv          # 19 + 18 = 37
ppo = PPO(state_dim, m.nu)       # 12 actuators

CKPT = "ppo_a1.pth"
if os.path.exists(CKPT):
    ppo.load(CKPT)
    print(f"Loaded checkpoint {CKPT}")
ppo.policy.to(device)

# ── Training loop ─────────────────────────────────────────────────────────────
steps_per_rollout = 2048
max_updates       = 5000   # ~10M steps

viewer = mujoco.viewer.launch_passive(m, d)

for update in range(max_updates):
    states, actions, log_probs = [], [], []
    rewards, dones, values     = [], [], []

    for step in range(steps_per_rollout):
        raw_obs = np.concatenate([d.qpos, d.qvel])
        state   = torch.tensor(ppo.normalize_obs(raw_obs), dtype=torch.float32).to(device)

        with torch.no_grad():
            action, log_prob, state_value = ppo.policy(state)

        ctrl = torch.tanh(action).cpu().numpy()
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)

        # NaN guard
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            mujoco.mj_resetData(m, d)
            rewards.append(0.0); dones.append(1.0)
            states.append(state); actions.append(action)
            log_probs.append(log_prob); values.append(state_value.squeeze())
            viewer.sync()
            continue

        height   = d.qpos[2]                                    # trunk z height
        is_alive = A1_ALIVE_MIN < height < A1_ALIVE_MAX

        forward_vel = d.qvel[0]                                 # x velocity
        alive_bonus = 2.0 if is_alive else 0.0                  # lower than humanoid — A1 is easier to keep upright
        ctrl_cost   = 0.001 * np.sum(np.square(d.ctrl))
        reward      = alive_bonus + (forward_vel if is_alive else 0.0) - ctrl_cost
        done        = not is_alive

        states.append(state);  actions.append(action)
        log_probs.append(log_prob); rewards.append(reward)
        dones.append(float(done)); values.append(state_value.squeeze())
        viewer.sync()

        if done:
            mujoco.mj_resetData(m, d)

    ppo.update(states, actions, log_probs, rewards, dones, values)
    ppo.save(CKPT)

    total_r = sum(rewards)
    mean_r  = total_r / len(rewards)
    max_r   = max(rewards)
    episodes = int(sum(dones))
    print(f"[Update {update + 1:>4}/{max_updates}]  "
          f"mean_r={mean_r:+.3f}  "
          f"total_r={total_r:+.1f}  "
          f"max_r={max_r:+.3f}  "
          f"episodes={episodes}")
