import time
import random
import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
import mujoco

# Set identical global random seed to replicate dirt positions
SEED = 40
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

max_torque = 80.0
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
# 3. Dynamic Site & State Accessors (Version-Agnostic)
# ------------------------------------------------------------------
def get_eef_site_id(robot):
    """Dynamically finds the active end-effector site ID."""
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
    """Safely extracts current end-effector position vector."""
    eef_id = get_eef_site_id(robot)
    return robot.sim.data.site_xpos[eef_id].copy()

def get_eef_rotation_matrix(robot):
    """Safely extracts current end-effector 3x3 rotation matrix."""
    eef_id = get_eef_site_id(robot)
    return robot.sim.data.site_xmat[eef_id].reshape(3, 3).copy()

def get_eef_geoms(env, robot):
    """Get all geometries belonging to the active end-effector."""
    sim = env.sim
    eef_site_id = get_eef_site_id(robot)
    eef_geoms = set()
    
    # Method 1: Get all geoms from the end-effector body
    eef_body_id = sim.model.site_bodyid[eef_site_id]
    for g_id in range(sim.model.ngeom):
        if sim.model.geom_bodyid[g_id] == eef_body_id:
            eef_geoms.add(g_id)
    
    # Method 2: Get gripper geometries
    if hasattr(robot, "gripper") and hasattr(robot.gripper, "contact_geoms"):
        for g_name in robot.gripper.contact_geoms:
            if g_name in sim.model._geom_name2id:
                eef_geoms.add(sim.model.geom_name2id(g_name))
    
    # Method 3: Look up by name keywords
    for g_id in range(sim.model.ngeom):
        g_name = sim.model.geom_id2name(g_id)
        if g_name:
            g_name_lower = g_name.lower()
            if any(kw in g_name_lower for kw in ["wiper", "tool", "finger", "hand", "eef", "gripper", "attachment", "end_effector", "pad"]):
                eef_geoms.add(g_id)
    
    return eef_geoms

def get_contact_wrench_at_eef(env, robot):
    """Gets the raw contact wrench at the end-effector frame dynamically."""
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
    """Simulates a wrist F/T sensor and returns contact wrench in WORLD frame dynamically."""
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
    
    force_world = R_eef_to_world @ force_eef
    torque_world = R_eef_to_world @ torque_eef
    
    return np.concatenate([force_world, torque_world])

# ------------------------------------------------------------------
# 4. Force Controller Class (PI Controller)
# ------------------------------------------------------------------
class ForceController:
    """
    PI Force controller with saturation.
    """
    def __init__(self, 
                 desired_force=1.50,  # NEGATIVE = downward force!
                 kp_force=80.0,
                 ki_force=0.8,
                 max_force_correction=1.0,
                 max_integral=8.0,
                 force_deadzone=0.3
                 ):
        self.desired_force = desired_force
        self.kp = kp_force
        self.ki = ki_force
        self.max_force = max_force_correction
        self.max_integral = max_integral
        self.deadzone = force_deadzone
        
        self.integral_error = 0.0
        self.alpha = 0.3
        self.filtered_force = 0.0
        self.force_errors = []
        self.force_outputs = []
    
    def update(self, current_force_z, dt=0.005):
        """
        Update force controller with PI control.
        Returns force correction in Newtons.
        """
        # Apply low-pass filter to force measurement
        self.filtered_force = self.alpha * current_force_z + (1 - self.alpha) * self.filtered_force
        measured_force = self.filtered_force
        
        # Calculate force error (desired - actual)
        force_error = measured_force - self.desired_force
        
        # Apply deadzone
        if abs(force_error) < self.deadzone:
            force_error = 0.0
        
        # Update integral term with windup protection
        self.integral_error += force_error * dt
        self.integral_error = np.clip(self.integral_error, -self.max_integral, self.max_integral)
        
        # Compute force command (PI control)
        force_correction = (
            self.kp * force_error +
            self.ki * self.integral_error
        )
        
        # Apply saturation
        force_correction = np.clip(force_correction, -self.max_force, self.max_force)
        
        # Store for monitoring
        self.force_errors.append(force_error)
        self.force_outputs.append(force_correction)
        
        return force_correction
    
    def reset(self):
        """Reset controller state."""
        self.integral_error = 0.0
        self.filtered_force = 0.0
        self.force_errors = []
        self.force_outputs = []

# ------------------------------------------------------------------
# 5. Impedance Gains & Loggers
# ------------------------------------------------------------------
Kp_pos = np.diag([2000, 2000, 800])
Kp_ori = np.diag([800, 800, 800])

Kd_pos = 2 * np.sqrt(Kp_pos)
Kd_ori = 2 * np.sqrt(Kp_ori)

Kp = np.block([[Kp_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kp_ori]])
Kd = np.block([[Kd_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kd_ori]])

eef_id = get_eef_site_id(robot)
eef_name = robot.sim.model.site_id2name(eef_id)

# ------------------------------------------------------------------
# 6. Initialize Force Controller
# ------------------------------------------------------------------
force_ctrl = ForceController(
    desired_force=1.5, kp_force=80.0, ki_force=0.8,
    max_force_correction=1.0, max_integral=8.0, force_deadzone=0.3
)

desired_pos_list = []
actual_pos_list = []
desired_quat_list = []
actual_quat_list = []
ori_error_list = []
wrench_list = []
wrench_sensor_list = []
force_z_list = []
force_correction_list = []
force_error_list = []
position_offset_list = []

TOOL_MASS = 0.3
TOOL_COM_EEF = np.array([0, 0, 0.02])

# ------------------------------------------------------------------
# 7. Trajectory Replay Loop
# ------------------------------------------------------------------
step_idx = 0
print("\nStarting Force-Controlled Impedance Tracking Replay...")
print("="*110)
print(f"{'Step':>5} | {'EE Position (X, Y, Z) [m]':>28} | {'World Forces (Fx, Fy, Fz) [N]':>30} | {'World Torques (Tx, Ty, Tz) [Nm]':>30}")
print("="*110)

try:
    while step_idx < num_steps:
        x_desired = pos_traj[step_idx].copy()
        q_desired = quat_traj[step_idx].copy()

        # Safe, version-agnostic End-Effector pose extraction
        x_current = get_eef_position(robot)
        quat_current = T.mat2quat(get_eef_rotation_matrix(robot))
        dq = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()

        if np.dot(q_desired, quat_current) < 0:
            q_desired = -q_desired

        # Calculate Contact Wrench in World Frame
        ee_wrench_world = get_eef_wrench_from_ft_sensor(
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

        # Print EE Position & Wrench Output every 20 steps
        if step_idx % 20 == 0:
            pos_str = f"[{x_current[0]:.3f}, {x_current[1]:.3f}, {x_current[2]:.3f}]"
            force_str = f"[{ee_wrench_world[0]:.2f}, {ee_wrench_world[1]:.2f}, {ee_wrench_world[2]:.2f}]"
            torque_str = f"[{ee_wrench_world[3]:.3f}, {ee_wrench_world[4]:.3f}, {ee_wrench_world[5]:.3f}]"
            print(f"{step_idx:5d} | {pos_str:>28} | {force_str:>30} | {torque_str:>30}")

        # Force Control Update & Hybrid Impedance Control
        force_correction = force_ctrl.update(current_force_z, dt=0.005)
        force_correction_list.append(force_correction)
        force_error_list.append(force_ctrl.force_errors[-1] if force_ctrl.force_errors else 0)
        
        Kf = 0.2
        position_offset = Kf * force_correction 
        position_offset_list.append(position_offset)
        
        x_desired_modified = x_desired.copy()
        x_desired_modified[2] += position_offset
        
        max_position_offset = 0.3
        x_desired_modified[2] = np.clip(x_desired_modified[2], 
                                        x_desired[2] - max_position_offset, 
                                        x_desired[2] + max_position_offset)

        pos_error = x_desired_modified - x_current
        ori_error = T.get_orientation_error(q_desired, quat_current)
        cartesian_error = np.concatenate([pos_error, ori_error])

        desired_pos_list.append(x_desired.copy())
        actual_pos_list.append(x_current.copy())
        desired_quat_list.append(q_desired.copy())
        actual_quat_list.append(quat_current.copy())
        ori_error_list.append(ori_error.copy())

        J_pos = robot.sim.data.get_site_jacp(eef_name)[:, robot._ref_joint_vel_indexes]
        J_rot = robot.sim.data.get_site_jacr(eef_name)[:, robot._ref_joint_vel_indexes]
        J = np.vstack([J_pos, J_rot])
        dx = J @ dq

        F_task = Kp @ cartesian_error - Kd @ dx
        

        tau_task = J.T @ F_task
        tau_gravity = robot.sim.data.qfrc_bias[robot._ref_joint_vel_indexes].copy()

        tau_total = tau_task + tau_gravity
        normalized_action = np.clip(tau_total / max_torque, -1.0, 1.0)

        if env.action_dim > 7:
            action = np.concatenate([normalized_action, [-1.0]])
        else:
            action = normalized_action

        obs, reward, done, info = env.step(action)
        env.render()
        
        time.sleep(0.005)
        step_idx += 1

    print("="*110)
    print("Trajectory playback finished successfully!")

except KeyboardInterrupt:
    print("\nReplay stopped early by user.")

# ------------------------------------------------------------------
# 8. Plotting & Summary Output
# ------------------------------------------------------------------
if len(actual_pos_list) > 0:
    print("\nGenerating plots...")

    desired_pos = np.array(desired_pos_list)
    actual_pos = np.array(actual_pos_list)
    desired_quat = np.array(desired_quat_list)
    actual_quat = np.array(actual_quat_list)
    force_z = np.array(force_z_list)
    force_corr = np.array(force_correction_list)
    pos_offset = np.array(position_offset_list)
    wrench_sensor = np.array(wrench_sensor_list) if len(wrench_sensor_list) > 0 else None
    time_steps = np.arange(len(actual_pos)) * 0.005

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    fig.suptitle("Force-Controlled Impedance Control with Dynamic F/T Sensing", fontsize=14, fontweight="bold")

    # Plot 1: Position Tracking (X, Y, Z)
    pos_colors, pos_labels = ["crimson", "limegreen", "royalblue"], ["X", "Y", "Z"]
    for i in range(3):
        axes[0].plot(time_steps, desired_pos[:, i], "--", color=pos_colors[i], alpha=0.7, label=f"Desired {pos_labels[i]}")
        axes[0].plot(time_steps, actual_pos[:, i], "-", color=pos_colors[i], linewidth=1.5, label=f"Actual {pos_labels[i]}")
    axes[0].set_ylabel("Position [m]")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    # Plot 2: Quaternion Tracking
    quat_colors, quat_labels = ["darkorange", "crimson", "limegreen", "royalblue"], ["qw", "qx", "qy", "qz"]
    for i in range(4):
        axes[1].plot(time_steps, desired_quat[:, i], "--", color=quat_colors[i], alpha=0.7, label=f"Desired {quat_labels[i]}")
        axes[1].plot(time_steps, actual_quat[:, i], "-", color=quat_colors[i], linewidth=1.5, label=f"Actual {quat_labels[i]}")
    axes[1].set_ylabel("Quaternion")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)

    # Plot 3: Force Tracking (World Frame)
    axes[2].plot(time_steps, force_z, 'b-', linewidth=2, label='Measured Fz (World)')
    axes[2].axhline(y=force_ctrl.desired_force, color='r', linestyle='--', alpha=0.7, linewidth=2, label=f'Desired Force = {force_ctrl.desired_force:.1f}N')
    axes[2].fill_between(time_steps, force_ctrl.desired_force - force_ctrl.deadzone, force_ctrl.desired_force + force_ctrl.deadzone, color='g', alpha=0.15, label='Deadzone')
    axes[2].set_ylabel("Z Force [N]")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right", fontsize=10)
    axes[2].set_ylim([-25, 5])

    # Plot 4: Force Correction
    axes[3].plot(time_steps, force_corr, 'purple', linewidth=2, label='Force Correction')
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[3].set_ylabel("Force Correction [N]")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="upper right", fontsize=10)

    # Plot 5: Position Offset
    axes[4].plot(time_steps, pos_offset * 1000, 'teal', linewidth=2, label='Z Position Offset')
    axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[4].set_ylabel("Position Offset [mm]")
    axes[4].grid(True, linestyle="--", alpha=0.6)
    axes[4].legend(loc="upper right", fontsize=10)

    # Plot 6: Raw Sensor vs World Frame
    if wrench_sensor is not None:
        axes[5].plot(time_steps, wrench_sensor[:, 2], 'orange', linewidth=2, alpha=0.7, label='Raw Sensor Fz (EEF frame)')
        axes[5].plot(time_steps, force_z, 'b-', linewidth=2, label='World Frame Fz')
        axes[5].axhline(y=force_ctrl.desired_force, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Desired Force')
        axes[5].set_xlabel("Time [s]")
        axes[5].set_ylabel("Z Force [N]")
        axes[5].grid(True, linestyle="--", alpha=0.6)
        axes[5].legend(loc="upper right", fontsize=10)
        axes[5].set_ylim([-25, 5])

    plt.tight_layout()
    plt.savefig("force_controlled_wiping_ft_sensor.png", dpi=300)
    plt.show()