import mujoco
import mujoco.viewer
import time

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path("human.xml")
data  = mujoco.MjData(model)

print("Model loaded!")
print(f"  Bodies    : {model.nbody}")
print(f"  Joints    : {model.njnt}")
print(f"  DoFs      : {model.nv}")
print(f"  Actuators : {model.nu}")

# ---------------------------------------------------------------------------
# Map actuator index → (qpos_addr, qvel_addr) for hinge joints only.
# Ball joints store a quaternion in qpos and are handled by passive damping.
# Actuator order in XML: left_hip(ball), right_hip(ball),
#                        left_knee(hinge), right_knee(hinge),
#                        left_ankle(hinge), right_ankle(hinge)
# ---------------------------------------------------------------------------
HINGE_ACTUATORS = {
    # actuator_index : joint_name
    2: "left_knee",
    3: "right_knee",
    4: "left_ankle",
    5: "right_ankle",
}

hinge_map = {}
for act_idx, jname in HINGE_ACTUATORS.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0:
        hinge_map[act_idx] = (model.jnt_qposadr[jid], model.jnt_dofadr[jid])

KP = 20.0   # position gain
KD = 2.0    # velocity gain


def pd_control(model, data):
    """Hold hinge joints at zero angle (upright pose)."""
    data.ctrl[:] = 0.0
    for act_idx, (qp, qv) in hinge_map.items():
        q  = data.qpos[qp]
        qd = data.qvel[qv]
        data.ctrl[act_idx] = -KP * q - KD * qd


# ---------------------------------------------------------------------------
# Run with interactive viewer
# ---------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Nice default camera position
    viewer.cam.distance  = 4.0
    viewer.cam.elevation = -15
    viewer.cam.azimuth   = 90

    print("\nSimulation running — close the viewer window to stop.")

    while viewer.is_running():
        step_start = time.time()

        pd_control(model, data)
        mujoco.mj_step(model, data)

        viewer.sync()

        # Pace to real time
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)

print("Simulation finished.")
