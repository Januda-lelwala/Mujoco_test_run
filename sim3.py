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
from rl import ActorCritic, PPO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Load model
m = mujoco.MjModel.from_xml_path(mujoco_humanoid_mj_description.MJCF_PATH)
d = mujoco.MjData(m)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

actorcritic = ActorCritic(m.nq, m.nu).to(device)
ppo = PPO(m.nq, m.nu)



# =============================
# Training Loop
# =============================
steps_per_rollout = 2048
max_updates = 1000

viewer = mujoco.viewer.launch_passive(m, d)

for update in range(max_updates):
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    for step in range(steps_per_rollout):
        state = torch.tensor(d.qpos, dtype=torch.float32).to(device)
        action_probs, state_value = actorcritic(state)

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        d.ctrl[:] = 0.0
        for i in range(m.nu):
            d.ctrl[i] = action_probs[i].item() * 2 - 1  # Scale to [-1, 1]

        mujoco.mj_step(m, d)

        # Walking reward
        forward_vel = d.qvel[0]                              # reward forward (x) velocity
        height = d.qpos[2]                                   # torso height
        alive = 1.0 if 0.8 < height < 2.1 else 0.0          # bonus for staying upright
        ctrl_cost = 0.001 * np.sum(np.square(d.ctrl))        # penalty for large actuations
        reward = forward_vel + alive - ctrl_cost
        done = height < 0.8 or height > 2.1                  # episode ends if humanoid falls

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)
        values.append(state_value)
        viewer.sync()

    # Compute advantages and returns, then update policy with PPO
    ppo.update(states, actions, log_probs, rewards, dones, values)
    print(f"Update {update + 1}/{max_updates} completed.")
