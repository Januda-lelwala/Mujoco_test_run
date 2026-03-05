import sys
import os
import subprocess

# On macOS, launch_passive requires mjpython. Re-launch under it if needed.
# Use an env-var sentinel so we don't re-launch infinitely.
if sys.platform == "darwin" and os.environ.get("_MJPYTHON_RELAUNCHED") != "1":
    mjpython = os.path.join(os.path.dirname(sys.executable), "mjpython")
    env = os.environ.copy()
    env["_MJPYTHON_RELAUNCHED"] = "1"
    sys.exit(subprocess.call([mjpython, __file__] + sys.argv[1:], env=env))

import time
import numpy as np
import mujoco
import mujoco.viewer
from robot_descriptions import mujoco_humanoid_mj_description
from rl import PPO
import torch


# Load model
m = mujoco.MjModel.from_xml_path(mujoco_humanoid_mj_description.MJCF_PATH)
d = mujoco.MjData(m)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# state = qpos + qvel; use a single network owned by PPO
state_dim = m.nq + m.nv
ppo = PPO(state_dim, m.nu)

if os.path.exists("ppo_humanoid_update.pth"):
    ppo.load("ppo_humanoid_update.pth")
ppo.policy.to(device)

# =============================
# Training Loop
# =============================
steps_per_rollout = 2048
max_updates = 5000  # ~10M steps — minimum to see real progress on humanoid

viewer = mujoco.viewer.launch_passive(m, d)

for update in range(max_updates):
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    for step in range(steps_per_rollout):
        raw_obs = np.concatenate([d.qpos, d.qvel])
        state = torch.tensor(
            ppo.normalize_obs(raw_obs), dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            action, log_prob, state_value = ppo.policy(state)

        # Tanh squashes raw action to [-1, 1] matching actuator range
        ctrl = torch.tanh(action).cpu().numpy()
        d.ctrl[:] = ctrl

        mujoco.mj_step(m, d)

        # --- NaN guard: unstable sim poisons gradients, reset immediately ---
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            mujoco.mj_resetData(m, d)
            rewards.append(0.0)
            dones.append(1.0)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(state_value.squeeze())
            viewer.sync()
            continue

        # Walking reward — alive gates everything so standing comes first
        forward_vel = d.qvel[0]                              # reward forward (x) velocity
        height = d.qpos[2]                                   # torso height
        is_alive = 0.8 < height < 2.1
        alive_bonus = 5.0 if is_alive else 0.0               # strong incentive to stay upright
        ctrl_cost = 0.001 * np.sum(np.square(d.ctrl))        # penalty for large actuations
        # forward_vel only counts when alive, preventing reward from tumbling forward
        reward = alive_bonus + (forward_vel if is_alive else 0.0) - ctrl_cost
        done = not is_alive                                  # episode ends if humanoid falls

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(float(done))
        values.append(state_value.squeeze())
        viewer.sync()

        if done:
            mujoco.mj_resetData(m, d)                        # reset sim on episode end

    # Compute advantages and returns, then update policy with PPO
    ppo.update(states, actions, log_probs, rewards, dones, values)
    ppo.save("ppo_humanoid_update.pth")

    total_reward = sum(rewards)
    mean_reward = total_reward / len(rewards)
    max_reward = max(rewards)
    episodes = int(sum(dones))
    print(f"[Update {update + 1:>4}/{max_updates}]  "
          f"mean_r={mean_reward:+.3f}  "
          f"total_r={total_reward:+.1f}  "
          f"max_r={max_reward:+.3f}  "
          f"episodes={episodes}")
