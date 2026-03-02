# import sys
# import os
# import subprocess

# # On macOS, launch_passive requires mjpython. Re-launch under it if needed.
# if sys.platform == "darwin" and os.path.basename(sys.executable) != "mjpython":
#     mjpython = os.path.join(os.path.dirname(sys.executable), "mjpython")
#     sys.exit(subprocess.call([mjpython, __file__] + sys.argv[1:]))

import mujoco
import mujoco.viewer

# Loading a specific model description as an imported module.
from robot_descriptions import g1_description
m = mujoco.MjModel.from_xml_path(g1_description.URDF_PATH)
d = mujoco.MjData(m)

mujoco.viewer.launch(m, d)