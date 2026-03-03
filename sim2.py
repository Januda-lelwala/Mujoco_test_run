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

# Load model
m = mujoco.MjModel.from_xml_path(mujoco_humanoid_mj_description.MJCF_PATH)
d = mujoco.MjData(m)

# --- Actuator control demo ---
# Sinusoidal motion on all joints, each with a different phase.

AMPLITUDE = 0.3   # radians
FREQUENCY = 0.3   # Hz

PHASES = [i * (np.pi / m.nu) for i in range(m.nu)]

def set_ctrl(data, t):
    for i in range(m.nu):
        lo, hi = m.actuator_ctrlrange[i]
        mid = (lo + hi) / 2
        data.ctrl[i] = mid + AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * t + PHASES[i])

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        set_ctrl(d, d.time)       # apply control signals
        mujoco.mj_step(m, d)      # step physics

        viewer.sync()

        # Pace simulation to real time
        elapsed = time.time() - step_start
        remaining = m.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)
