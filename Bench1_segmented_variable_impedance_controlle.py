import time
import random
import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
import mujoco

# Set identical global random seed to replicate dirt positions
SEED = 15
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------------------
# 0. Load Recorded Trajectory
# ------------------------------------------------------------------
try:
    data = np.load("auto_wiping_trajectory.npz")
    pos_traj = data["pos"]        # (N, 3)
    quat_traj = data["quat"]      # (N, 4) in [w, x, y, z] format
    num_steps = len(pos_traj)
    print(f"Successfully loaded {num_steps} trajectory frames.")
except FileNotFoundError:
    print("Error: 'auto_wiping_trajectory.npz' not found!")
    exit()

# ------------------------------------------------------------------
# 1. Controller Setup (Torque Control)
# ------------------------------------------------------------------
controller_config = load_composite_controller_config(controller="BASIC")

max_torque = 50.0
controller_config["body_parts"]["right"]["type"] = "JOINT_TORQUE"
controller_config["body_parts"]["right"]["input_max"] = [max_torque] * 7
controller_config["body_parts"]["right"]["input_min"] = [-max_torque] * 7
controller_config["body_parts"]["right"]["output_max"] = [max_torque] * 7
controller_config["body_parts"]["right"]["output_min"] = [-max_torque] * 7

# ------------------------------------------------------------------
# 2. Environment Setup
# ------------------------------------------------------------------
env = suite.make(
    env_name="Wipe",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
    horizon=1000000,
    seed=SEED,
)

obs = env.reset()
robot = env.robots[0]

# Enable contact visualization
env.sim.model.viscontype = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
env.sim.model.visconsize = 0.01
env.sim.model.visconrgb = np.array([1, 0, 0])

# ------------------------------------------------------------------
# 3. Dynamic Site & State Accessors
# ------------------------------------------------------------------
def get_eef_site_id(robot):
    sim = robot.sim
    possible_names = ["eef_site", "gripper_site", "right_eef", "right_hand"]
    
    if hasattr(robot, "eef_site_id"):
        if isinstance(robot.eef_site_id, dict):
            if "right" in robot.eef_site_id:
                return robot.eef_site_id["right"]
        elif isinstance(robot.eef_site_id, (int, np.integer)):
            return robot.eef_site_id

    for name in possible_names:
        if name in sim.model.site_names:
            return sim.model.site_name2id(name)

    return robot.eef_site_id["right"]

def get_eef_position(robot):
    eef_id = get_eef_site_id(robot)
    return robot.sim.data.site_xpos[eef_id].copy()

def get_eef_rotation_matrix(robot):
    eef_id = get_eef_site_id(robot)
    return robot.sim.data.site_xmat[eef_id].reshape(3, 3).copy()

def get_eef_geoms(env, robot):
    sim = env.sim
    eef_site_id = get_eef_site_id(robot)
    eef_geoms = set()
    
    eef_body_id = sim.model.site_bodyid[eef_site_id]
    for g_id in range(sim.model.ngeom):
        if sim.model.geom_bodyid[g_id] == eef_body_id:
            eef_geoms.add(g_id)
    
    if hasattr(robot, "gripper") and hasattr(robot.gripper, "contact_geoms"):
        for g_name in robot.gripper.contact_geoms:
            if g_name in sim.model._geom_name2id:
                eef_geoms.add(sim.model.geom_name2id(g_name))
    
    for g_id in range(sim.model.ngeom):
        g_name = sim.model.geom_id2name(g_id)
        if g_name:
            g_name_lower = g_name.lower()
            if any(kw in g_name_lower for kw in ["wiper", "tool", "finger", "hand", "eef", "gripper", "attachment", "end_effector", "pad"]):
                eef_geoms.add(g_id)
    
    return eef_geoms

def get_contact_wrench_at_eef(env, robot):
    sim = env.sim
    eef_site_id = get_eef_site_id(robot)
    eef_pos = sim.data.site_xpos[eef_site_id]
    eef_rotmat = sim.data.site_xmat[eef_site_id].reshape(3, 3)
    
    eef_geoms = get_eef_geoms(env, robot)
    
    total_force = np.zeros(3)
    total_torque = np.zeros(3)
    
    num_contacts = sim.data.ncon
    
    for i in range(num_contacts):
        contact = sim.data.contact[i]
        
        is_geom1 = contact.geom1 in eef_geoms
        is_geom2 = contact.geom2 in eef_geoms
        
        if is_geom1 or is_geom2:
            c_force = np.zeros(6)
            mujoco.mj_contactForce(sim.model._model, sim.data._data, i, c_force)
            c_frame = contact.frame.reshape(3, 3)
            force_world = c_frame.T @ c_force[:3]
            
            if is_geom2:
                force_world = -force_world
            
            force_eef = eef_rotmat.T @ force_world
            r_eef = contact.pos - eef_pos
            r_eef_local = eef_rotmat.T @ r_eef
            torque_eef = np.cross(r_eef_local, force_eef)
            
            total_force += force_eef
            total_torque += torque_eef
    
    return np.concatenate([total_force, total_torque])

def add_sensor_noise(wrench, force_noise_std=0.05, torque_noise_std=0.005):
    noise = np.concatenate([
        np.random.normal(0, force_noise_std, 3),
        np.random.normal(0, torque_noise_std, 3)
    ])
    return wrench + noise

def compensate_gravity_tool(wrench_sensor, tool_mass, tool_com_eef, R_eef_to_world):
    gravity_world = np.array([0, 0, -9.81])
    force_gravity_world = tool_mass * gravity_world
    
    force_gravity_eef = R_eef_to_world.T @ force_gravity_world
    torque_gravity_eef = np.cross(tool_com_eef, force_gravity_eef)
    
    wrench_compensated = wrench_sensor.copy()
    wrench_compensated[:3] -= force_gravity_eef
    wrench_compensated[3:] -= torque_gravity_eef
    
    return wrench_compensated

def get_eef_wrench_from_ft_sensor(env, robot, 
                                  tool_mass=0.3, 
                                  tool_com_eef=np.zeros(3),
                                  add_noise=True,
                                  force_noise_std=0.05,
                                  torque_noise_std=0.005):
    R_eef_to_world = get_eef_rotation_matrix(robot)
    contact_wrench_eef = get_contact_wrench_at_eef(env, robot)
    
    gravity_world = np.array([0, 0, -9.81])
    force_gravity_eef = R_eef_to_world.T @ (tool_mass * gravity_world)
    torque_gravity_eef = np.cross(tool_com_eef, force_gravity_eef)
    
    raw_ft_sensor_reading = contact_wrench_eef.copy()
    raw_ft_sensor_reading[:3] += force_gravity_eef
    raw_ft_sensor_reading[3:] += torque_gravity_eef
    
    if add_noise:
        raw_ft_sensor_reading = add_sensor_noise(raw_ft_sensor_reading, force_noise_std, torque_noise_std)
    
    wrench_compensated = compensate_gravity_tool(
        raw_ft_sensor_reading, tool_mass, tool_com_eef, R_eef_to_world
    )
    
    force_eef = -wrench_compensated[:3]
    torque_eef = -wrench_compensated[3:]
    ee_wrench_tool = np.concatenate([force_eef, torque_eef])
    
    force_world = R_eef_to_world @ force_eef
    torque_world = R_eef_to_world @ torque_eef
    ee_wrench_world = np.concatenate([force_world, torque_world])
    
    return ee_wrench_world, ee_wrench_tool

def get_joint_dynamics(robot):
    qpos = robot.sim.data.qpos[robot._ref_joint_pos_indexes]
    qvel = robot.sim.data.qvel[robot._ref_joint_vel_indexes]
    
    M = np.zeros((robot.dof, robot.dof))
    mujoco.mj_fullM(robot.sim.model._model, M, robot.sim.data.qM)
    M = M[np.ix_(robot._ref_joint_vel_indexes, robot._ref_joint_vel_indexes)]
    
    h = robot.sim.data.qfrc_bias[robot._ref_joint_vel_indexes].copy()
    return M, h

# ------------------------------------------------------------------
# 4. ART-AVIC Controller Class (Wang et al., 2027)
# ------------------------------------------------------------------
class ART_AVIC_Controller:
    """
    Adaptive Reference Trajectory & Adaptive Variable Impedance Controller (Wang et al., 2027)
    
    Implements:
    - Eq. (6) Reference trajectory adaptive adjustment: xr(t) = xr(t-T) + kx * (fd - fe)
    - Eq. (11)-(13) Segmented damping adaptation:
        Δb = (fe - fd) * θ * α1 * b^2   if (fd - fe) < 0  (Overshoot/High contact force)
        Δb = (fe - fd) * θ * α2 * b     if (fd - fe) >= 0 (Undershoot/Approach stage)
    - Eq. (11) Implemented via Cartesian Impedance command to operational space execution.
    """
    def __init__(self,
                 desired_force=2.0,
                 m_val=1.0,
                 b_init=40.0,
                 k_val=400.0,
                 kf_val=1.0,
                 kx_val=1e-4,
                 theta_val=0.01,
                 alpha1=0.001,
                 alpha2=0.01,
                 b_min=10.0,
                 b_max=300.0,
                 alpha_filter=0.3):
        
        self.fd = desired_force
        self.m = m_val
        self.b_base = b_init
        self.b_curr = b_init
        self.k = k_val
        self.kf = kf_val
        self.kx = kx_val
        
        # Damping adaptation hyperparameters
        self.theta = theta_val
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.b_min = b_min
        self.b_max = b_max
        
        self.alpha_filter = alpha_filter
        self.filtered_force = 0.0
        
        # State tracking for trajectory correction along normal
        self.xr_normal_offset = 0.0

    def update(self, current_force_world, normal_vec, dt=0.005):
        # 1. Projected scalar contact force fe along wiping normal n
        n = normal_vec / (np.linalg.norm(normal_vec) + 1e-8)
        fe_raw = float(np.dot(current_force_world, n))
        
        # Force low-pass filtering
        fe = self.alpha_filter * fe_raw + (1.0 - self.alpha_filter) * self.filtered_force
        self.filtered_force = fe
        
        # Force error e_f = fd - fe
        e_f = self.fd - fe
        
        # 2. Reference Trajectory Adaptive Adjustment Law - Eq. (6)
        # xr_offset accumulation along normal direction: Δxr = kx * (fd - fe)
        delta_xr = self.kx * e_f
        self.xr_normal_offset += delta_xr
        
        # 3. Segmented Damping Update Law - Eq. (12)
        # Note: (fe - fd) = -e_f
        if e_f < 0:
            # Over-force stage (fd - fe < 0)
            delta_b = (fe - self.fd) * self.theta * self.alpha1 * (self.b_curr ** 2)
        else:
            # Under-force / Approach stage (fd - fe >= 0)
            delta_b = (fe - self.fd) * self.theta * self.alpha2 * self.b_curr
            
        # Update actual damping parameter B = b + Δb - Eq. (13)
        self.b_curr = np.clip(self.b_curr + delta_b, self.b_min, self.b_max)
        
        pos_offset_3d = -self.xr_normal_offset * n
        return pos_offset_3d, self.b_curr, fe, e_f

# ------------------------------------------------------------------
# 5. Initialize Controller & Loggers
# ------------------------------------------------------------------
controller = ART_AVIC_Controller(
    desired_force=2.0,
    m_val=1.0,
    b_init=50.0,
    k_val=500.0,
    kf_val=1.0,
    kx_val=1.5e-4,       # Reference trajectory adaptation gain
    theta_val=0.005,      # Damping update rate
    alpha1=0.002,         # Damping update param 1
    alpha2=0.02,          # Damping update param 2
    b_min=10.0,
    b_max=300.0
)

desired_force_list = []
desired_pos_list = []
actual_pos_list = []
desired_quat_list = []
actual_quat_list = []
ori_error_list = []
wrench_list = []
wrench_sensor_list = []
force_z_list = []
force_error_list = []
position_offset_list = []
damping_b_list = []

tau_task_list = []
tau_total_list = []
tau_joints_list = []

TOOL_MASS = 0.3
TOOL_COM_EEF = np.array([0, 0, 0.02])

# Base Cartesian Impedance Gains
Kp_pos_base = np.array([2000.0, 2000.0, 2000.0])
Kp_ori_base = np.array([500.0, 500.0, 500.0])
Kd_ori_base = 2.0 * np.sqrt(Kp_ori_base)

# ------------------------------------------------------------------
# 6. Trajectory Replay Loop
# ------------------------------------------------------------------
eef_id = get_eef_site_id(robot)
eef_name = robot.sim.model.site_id2name(eef_id)

step_idx = 0
dt_step = 0.005

print("\nStarting Benchmark Replay (Wang et al. ART-AVIC)...")
print("="*110)
print(f"{'Step':>5} | {'EE Position (X, Y, Z) [m]':>28} | {'World Forces (Fx, Fy, Fz) [N]':>30} | {'Damping B [N*s/m]':>20}")
print("="*110)

try:
    while step_idx < num_steps:
        x_desired = pos_traj[step_idx].copy()
        q_desired = quat_traj[step_idx].copy()

        x_current = get_eef_position(robot)
        quat_current = T.mat2quat(get_eef_rotation_matrix(robot))
        dq = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()

        if np.dot(q_desired, quat_current) < 0:
            q_desired = -q_desired

        # Read FT sensor wrench
        ee_wrench_world, ee_wrench_tool = get_eef_wrench_from_ft_sensor(
            env, robot,
            tool_mass=TOOL_MASS,
            tool_com_eef=TOOL_COM_EEF,
            add_noise=True,
            force_noise_std=0.05,
            torque_noise_std=0.005
        )
        wrench_list.append(ee_wrench_world.copy())
        
        raw_sensor_wrench = get_contact_wrench_at_eef(env, robot)
        wrench_sensor_list.append(raw_sensor_wrench.copy())

        current_force_z = ee_wrench_world[2]
        force_z_list.append(current_force_z)

        # ----------------------------------------------------------
        # 1. Update ART-AVIC Trajectory & Damping
        # ----------------------------------------------------------
        R_s = T.quat2mat(q_desired)
        normal_vec = -R_s[:, 2]  # Tool z-axis pointing into wiping surface
        
        ee_force_world = ee_wrench_world[:3]
        
        pos_offset_3d, b_adapted, fe_filt, force_err = controller.update(
            current_force_world=ee_force_world,
            normal_vec=normal_vec,
            dt=dt_step
        )

        desired_force_list.append(controller.fd)
        force_error_list.append(force_err)
        position_offset_list.append(np.linalg.norm(pos_offset_3d))
        damping_b_list.append(b_adapted)

        # Apply Reference Trajectory Correction Eq. (6)
        x_desired_modified = x_desired + pos_offset_3d

        # ----------------------------------------------------------
        # 2. Operational Space Impedance Torque Computation
        # ----------------------------------------------------------
        ori_error = T.get_orientation_error(q_desired, quat_current)
        pos_error = x_desired_modified - x_current
        cartesian_error = np.concatenate([pos_error, ori_error])

        desired_pos_list.append(x_desired.copy())
        actual_pos_list.append(x_current.copy())
        desired_quat_list.append(q_desired.copy())
        actual_quat_list.append(quat_current.copy())
        ori_error_list.append(ori_error.copy())

        # Anisotropic Damping Matrix (b_adapted along contact normal, base elsewhere)
        n_sq = (normal_vec / (np.linalg.norm(normal_vec) + 1e-8)) ** 2
        Kd_pos_vec = (1.0 - n_sq) * (2.0 * np.sqrt(Kp_pos_base)) + n_sq * b_adapted
        
        Kp_pos = np.diag(Kp_pos_base)
        Kd_pos = np.diag(Kd_pos_vec)
        Kp_ori = np.diag(Kp_ori_base)
        Kd_ori = np.diag(Kd_ori_base)

        Kp = np.block([[Kp_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kp_ori]])
        Kd = np.block([[Kd_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kd_ori]])

        # Joint dynamics & Task Space Mapping
        J_pos = robot.sim.data.get_site_jacp(eef_name)[:, robot._ref_joint_vel_indexes]
        J_rot = robot.sim.data.get_site_jacr(eef_name)[:, robot._ref_joint_vel_indexes]
        J = np.vstack([J_pos, J_rot])
        dx = J @ dq

        M_q, h_q = get_joint_dynamics(robot)

        # Nullspace damping projection
        N = np.eye(robot.dof) - np.linalg.pinv(J) @ J
        tau_null = N @ (-3.0 * dq)

        # Force Error Gain integration kf * (fd - fe) into impedance law Eq. (11)
        F_force_feedforward = controller.kf * force_err * (normal_vec / (np.linalg.norm(normal_vec) + 1e-8))
        
        F_task = Kp @ cartesian_error - Kd @ dx
        F_task[:3] += F_force_feedforward

        tau_task = J.T @ F_task
        tau_total = tau_task + h_q + tau_null
        tau_total = np.clip(tau_total, -max_torque, max_torque)

        tau_task_list.append(np.linalg.norm(tau_task))
        tau_total_list.append(np.linalg.norm(tau_total))
        tau_joints_list.append(tau_total.copy())

        if step_idx % 20 == 0:
            pos_str = f"[{x_current[0]:.3f}, {x_current[1]:.3f}, {x_current[2]:.3f}]"
            force_str = f"[{ee_wrench_world[0]:.2f}, {ee_wrench_world[1]:.2f}, {ee_wrench_world[2]:.2f}]"
            print(f"{step_idx:5d} | {pos_str:>28} | {force_str:>30} | Damping B: {b_adapted:18.2f}")

        normalized_action = np.clip(tau_total / max_torque, -1.0, 1.0)
        action = np.concatenate([normalized_action, [-1.0]]) if env.action_dim > 7 else normalized_action

        obs, reward, done, info = env.step(action)
        env.render()

        time.sleep(0.005)
        step_idx += 1

    print("="*110)
    print("ART-AVIC Trajectory playback finished successfully!")

except KeyboardInterrupt:
    print("\nReplay stopped early by user.")

# ------------------------------------------------------------------
# 7. Plotting Benchmark Results
# ------------------------------------------------------------------
try:
    env.close()
    if hasattr(env, "viewer") and env.viewer is not None:
        env.viewer = None
except Exception:
    pass

if len(actual_pos_list) > 0:
    print("\nGenerating benchmark plots...")

    desired_pos = np.array(desired_pos_list)
    actual_pos = np.array(actual_pos_list)
    desired_quat = np.array(desired_quat_list)
    actual_quat = np.array(actual_quat_list)
    force_z = np.array(force_z_list)
    desired_force_arr = np.array(desired_force_list)
    pos_offset = np.array(position_offset_list)
    b_arr = np.array(damping_b_list)
    tau_joints_arr = np.atleast_2d(np.array(tau_joints_list))  

    time_steps = np.arange(len(actual_pos)) * dt_step

    fig1, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
    
    pos_colors, pos_labels = ["crimson", "limegreen", "royalblue"], ["X", "Y", "Z"]
    for i in range(3):
        axes[0].plot(time_steps, desired_pos[:, i], "--", color=pos_colors[i], alpha=0.7, label=f"Desired {pos_labels[i]}")
        axes[0].plot(time_steps, actual_pos[:, i], "-", color=pos_colors[i], linewidth=1.5, label=f"Actual {pos_labels[i]}")
    axes[0].set_ylabel("Position [m]")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    quat_colors, quat_labels = ["darkorange", "crimson", "limegreen", "royalblue"], ["qw", "qx", "qy", "qz"]
    for i in range(4):
        axes[1].plot(time_steps, desired_quat[:, i], "--", color=quat_colors[i], alpha=0.7, label=f"Desired {quat_labels[i]}")
        axes[1].plot(time_steps, actual_quat[:, i], "-", color=quat_colors[i], linewidth=1.5, label=f"Actual {quat_labels[i]}")
    axes[1].set_ylabel("Quaternion")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)

    axes[2].plot(time_steps, force_z, "b-", label="Measured Fz (ART-AVIC)")
    axes[2].plot(time_steps, desired_force_arr, "r--", label="Target Force (2.0 N)")
    axes[2].set_ylabel("Z Force [N]")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(time_steps, pos_offset * 1000, "teal", label="Eq. (6) Trajectory Correction [mm]")
    axes[3].set_ylabel("Traj Offset [mm]")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="upper right", fontsize=8)

    axes[4].plot(time_steps, b_arr, "purple", label="Eq. (12) Segmented Damping B [N*s/m]")
    axes[4].set_xlabel("Time [s]")
    axes[4].set_ylabel("Damping B")
    axes[4].grid(True, linestyle="--", alpha=0.6)
    axes[4].legend(loc="upper right", fontsize=8)

    fig1.tight_layout()
    fig1.savefig("art_avic_wiping_results.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    joint_colors = ["crimson", "darkorange", "goldenrod", "limegreen", "teal", "royalblue", "purple"]

    num_joints = tau_joints_arr.shape[1] if tau_joints_arr.ndim > 1 else 0
    for j in range(num_joints):
        ax2.plot(time_steps[:len(tau_joints_arr)], tau_joints_arr[:, j], color=joint_colors[j % len(joint_colors)], linewidth=1.2, label=f"Joint {j+1}")

    ax2.axhline(y=max_torque, color="r", linestyle="--", alpha=0.6, label=f"±{max_torque} Nm Limit")
    ax2.axhline(y=-max_torque, color="r", linestyle="--", alpha=0.6)
    ax2.set_title("7-DoF Franka Panda Joint Control Torques (ART-AVIC Benchmark)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Torque [Nm]")
    ax2.set_ylim(-max_torque * 1.1, max_torque * 1.1)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", ncol=4, fontsize=8)

    fig2.tight_layout()
    fig2.savefig("art_avic_joint_torques.png", dpi=300)

    plt.show(block=True)