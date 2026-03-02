import mujoco, sys

model = mujoco.MjModel.from_xml_path("human.xml")
data  = mujoco.MjData(model)

HINGE_ACTUATORS = {2:"left_knee", 3:"right_knee", 4:"left_ankle", 5:"right_ankle"}
hinge_map = {}
for idx, jname in HINGE_ACTUATORS.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    hinge_map[idx] = (model.jnt_qposadr[jid], model.jnt_dofadr[jid])

KP, KD = 20.0, 2.0

def pd(model, data):
    data.ctrl[:] = 0.0
    for i, (qp, qv) in hinge_map.items():
        data.ctrl[i] = -KP * data.qpos[qp] - KD * data.qvel[qv]

for _ in range(2000):
    pd(model, data)
    mujoco.mj_step(model, data)

print(f"2000 steps ({2000 * model.opt.timestep:.1f}s) OK  |  torso_z = {data.qpos[2]:.3f} m")
