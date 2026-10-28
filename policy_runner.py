"""
Standalone Policy Rollout Script for Wiping Task
Saves trained policy parameters and allows replay with visualization
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
import mujoco
import pickle
import os
from pathlib import Path

# Set identical global random seed
SEED = 40
np.random.seed(SEED)
random.seed(SEED)

# ============================================================================
# CONSTANTS
# ============================================================================
MAX_EVAL_STEPS = 8000
TOOL_MASS = 0.3
TOOL_COM_EEF = np.array([0, 0, 0.02])
max_torque = 80.0
POLICY_DIR = "trained_policies"
os.makedirs(POLICY_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD TRAJECTORY
# ============================================================================
try:
    data = np.load("auto_wiping_trajectory.npz")
    pos_traj = data["pos"]
    quat_traj = data["quat"]
    num_steps = len(pos_traj)
    print(f"✓ Loaded trajectory with {num_steps} frames")
except FileNotFoundError:
    print("✗ Error: 'auto_wiping_trajectory.npz' not found!")
    exit()

# ============================================================================
# 2. CONTROLLER SETUP
# ============================================================================
controller_config = load_composite_controller_config(controller="BASIC")
controller_config["body_parts"]["right"]["type"] = "JOINT_TORQUE"
controller_config["body_parts"]["right"]["input_max"] = [max_torque] * 7
controller_config["body_parts"]["right"]["input_min"] = [-max_torque] * 7
controller_config["body_parts"]["right"]["output_max"] = [max_torque] * 7
controller_config["body_parts"]["right"]["output_min"] = [-max_torque] * 7

# ============================================================================
# 3. UTILITY FUNCTIONS (Same as original)
# ============================================================================
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
                eef_geoms.add(sim.model.geom_name2id(g_name))
    
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
    
    force_world = R_eef_to_world @ force_eef
    torque_world = R_eef_to_world @ torque_eef
    
    return np.concatenate([force_world, torque_world])

# ============================================================================
# 4. FORCE CONTROLLER CLASS
# ============================================================================
class ForceController:
    def __init__(self, 
                 desired_force=1.5,
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
        self.filtered_force = self.alpha * current_force_z + (1 - self.alpha) * self.filtered_force
        measured_force = self.filtered_force
        
        force_error = measured_force - self.desired_force
        
        if abs(force_error) < self.deadzone:
            force_error = 0.0
        
        self.integral_error += force_error * dt
        self.integral_error = np.clip(self.integral_error, -self.max_integral, self.max_integral)
        
        force_correction = (
            self.kp * force_error +
            self.ki * self.integral_error
        )
        
        force_correction = np.clip(force_correction, -self.max_force, self.max_force)
        
        self.force_errors.append(force_error)
        self.force_outputs.append(force_correction)
        
        return force_correction
    
    def reset(self):
        self.integral_error = 0.0
        self.filtered_force = 0.0
        self.force_errors = []
        self.force_outputs = []

# ============================================================================
# 5. POLICY NETWORK CLASS
# ============================================================================
class PolicyNN:
    def __init__(self, input_dim=12, hidden_dim=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 7
        
        self.w1_shape = (input_dim, hidden_dim)
        self.b1_shape = (hidden_dim,)
        self.w2_shape = (hidden_dim, self.output_dim)
        self.b2_shape = (self.output_dim,)
        
        self.num_params = (input_dim * hidden_dim) + hidden_dim + (hidden_dim * self.output_dim) + self.output_dim
        
        # Initialize weights
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
    def set_params(self, params_vec):
        """Unflattens a 1D vector into weight matrices and bias vectors."""
        idx = 0
        w1_size = self.input_dim * self.hidden_dim
        self.W1 = params_vec[idx : idx + w1_size].reshape(self.w1_shape)
        idx += w1_size
        
        b1_size = self.hidden_dim
        self.b1 = params_vec[idx : idx + b1_size]
        idx += b1_size
        
        w2_size = self.hidden_dim * self.output_dim
        self.W2 = params_vec[idx : idx + w2_size].reshape(self.w2_shape)
        idx += w2_size
        
        self.b2 = params_vec[idx:]
    
    def get_params(self):
        """Flattens weights and biases into a 1D vector."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])
    
    def forward(self, state_vec):
        # 1. State Input Normalization
        pos_err_norm = state_vec[:3] / 0.05
        dq_norm = state_vec[3:10] / 1.0
        force_norm = (state_vec[10:11] - 1.5) / 3.0
        force_err_norm = state_vec[11:12] / 2.0
        
        normalized_state = np.concatenate([pos_err_norm, dq_norm, force_norm, force_err_norm])

        # 2. Forward Pass
        h = np.tanh(normalized_state @ self.W1 + self.b1)
        raw_out = h @ self.W2 + self.b2
        sig_out = 1.0 / (1.0 + np.exp(-raw_out))

        # 3. Dynamic Modulation
        pos_err_magnitude = np.linalg.norm(state_vec[:3])
        var_factor = 0.85 + 0.30 * np.tanh(pos_err_magnitude * 20.0)
        sig_modulated = np.clip(sig_out * var_factor, 0.0, 1.0)

        # 4. Final Gain Computation
        Kf = 0.05 + sig_modulated[0] * 0.15
        Kp_pos_diag = 1000.0 + sig_modulated[1:4] * 9000.0
        Kp_ori_diag = 200.0 + sig_modulated[4:7] * 1800.0

        return Kf, Kp_pos_diag, Kp_ori_diag

# ============================================================================
# 6. POLICY MANAGEMENT FUNCTIONS
# ============================================================================
def save_policy(policy_nn, filename="best_policy.pkl"):
    """Save the trained policy parameters to a file."""
    params = policy_nn.get_params()
    policy_data = {
        'params': params,
        'input_dim': policy_nn.input_dim,
        'hidden_dim': policy_nn.hidden_dim,
        'output_dim': policy_nn.output_dim,
        'num_params': policy_nn.num_params,
        'timestamp': time.strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    filepath = os.path.join(POLICY_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(policy_data, f)
    print(f"✓ Policy saved to: {filepath}")
    return filepath

def load_policy(filename="best_policy.pkl"):
    """Load a saved policy from file."""
    filepath = os.path.join(POLICY_DIR, filename)
    with open(filepath, 'rb') as f:
        policy_data = pickle.load(f)
    
    # Create policy network with saved architecture
    policy_nn = PolicyNN(
        input_dim=policy_data['input_dim'],
        hidden_dim=policy_data['hidden_dim']
    )
    policy_nn.set_params(policy_data['params'])
    
    print(f"✓ Policy loaded from: {filepath}")
    print(f"  - Input dim: {policy_data['input_dim']}")
    print(f"  - Hidden dim: {policy_data['hidden_dim']}")
    print(f"  - Num params: {policy_data['num_params']}")
    print(f"  - Saved: {policy_data.get('timestamp', 'unknown')}")
    
    return policy_nn

def list_saved_policies():
    """List all saved policy files."""
    files = [f for f in os.listdir(POLICY_DIR) if f.endswith('.pkl')]
    if files:
        print("\nAvailable policies:")
        for f in files:
            filepath = os.path.join(POLICY_DIR, f)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  - {f} ({size:.1f} KB)")
    else:
        print("No saved policies found.")
    return files

# ============================================================================
# 7. ROLLOUT FUNCTION
# ============================================================================
def run_rollout(policy_nn, 
                visualize=True, 
                record_data=False,
                plot_results=True,
                save_plot=True,
                max_steps=None,
                speed=0.005):
    """
    Run a single rollout with the given policy.
    
    Args:
        policy_nn: PolicyNN instance with loaded parameters
        visualize: Whether to show MuJoCo visualization
        record_data: Whether to record trajectory data
        plot_results: Whether to generate plots
        save_plot: Whether to save plots to file
        max_steps: Max number of steps (None for full trajectory)
        speed: Time delay between steps for visualization
    """
    # Create environment with rendering
    env = suite.make(
        env_name="Wipe",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=visualize,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        horizon=1000000,
        seed=SEED,
    )
    
    # Enable contact visualization
    env.sim.model.viscontype = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
    env.sim.model.visconsize = 0.01
    env.sim.model.visconrgb = np.array([1, 0, 0])
    
    obs = env.reset()
    robot = env.robots[0]
    eef_id = get_eef_site_id(robot)
    eef_name = robot.sim.model.site_id2name(eef_id)
    
    force_ctrl = ForceController(
        desired_force=1.5, kp_force=80.0, ki_force=0.8,
        max_force_correction=1.0, max_integral=8.0, force_deadzone=0.3
    )
    
    # Data recording
    if record_data:
        desired_pos_list, actual_pos_list = [], []
        force_z_list, force_correction_list = [], []
        position_offset_list, kf_list = [], []
        joint_positions, joint_velocities = [], []
    
    step_idx = 0
    limit_steps = min(num_steps, max_steps) if max_steps is not None else num_steps
    
    print(f"\n▶ Running rollout ({limit_steps} steps)...")
    start_time = time.time()
    
    while step_idx < limit_steps:
        x_desired = pos_traj[step_idx].copy()
        q_desired = quat_traj[step_idx].copy()
        
        x_current = get_eef_position(robot)
        quat_current = T.mat2quat(get_eef_rotation_matrix(robot))
        dq = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()
        
        if np.dot(q_desired, quat_current) < 0:
            q_desired = -q_desired
        
        ee_wrench_world = get_eef_wrench_from_ft_sensor(
            env, robot, tool_mass=TOOL_MASS, 
            tool_com_eef=TOOL_COM_EEF, add_noise=True
        )
        
        current_force_z = ee_wrench_world[2]
        force_correction = force_ctrl.update(current_force_z, dt=0.005)
        force_err = force_ctrl.force_errors[-1] if force_ctrl.force_errors else 0.0
        
        state_vec = np.concatenate([x_desired - x_current, dq, [current_force_z], [force_err]])
        Kf, Kp_pos_diag, Kp_ori_diag = policy_nn.forward(state_vec)
        
        # Print progress every 100 steps
        if step_idx % 100 == 0 and visualize:
            progress = (step_idx / limit_steps) * 100
            print(f"  [{progress:5.1f}%] Step {step_idx:4d} | Fz: {current_force_z:6.3f} N | Kf: {Kf:.4f}")
        
        # Assemble impedance matrices
        Kp_pos = np.diag(Kp_pos_diag)
        Kp_ori = np.diag(Kp_ori_diag)
        Kd_pos = 2 * np.sqrt(Kp_pos)
        Kd_ori = 2 * np.sqrt(Kp_ori)
        
        Kp = np.block([[Kp_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kp_ori]])
        Kd = np.block([[Kd_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kd_ori]])
        
        # Position offset
        position_offset = Kf * force_correction
        x_desired_modified = x_desired.copy()
        x_desired_modified[2] += position_offset
        
        max_position_offset = 0.05
        x_desired_modified[2] = np.clip(x_desired_modified[2], 
                                        x_desired[2] - max_position_offset, 
                                        x_desired[2] + max_position_offset)
        
        # Control computation
        pos_error = x_desired_modified - x_current
        ori_error = T.get_orientation_error(q_desired, quat_current)
        cartesian_error = np.concatenate([pos_error, ori_error])
        
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
        
        # Record data
        if record_data:
            desired_pos_list.append(x_desired.copy())
            actual_pos_list.append(x_current.copy())
            force_z_list.append(current_force_z)
            force_correction_list.append(force_correction)
            position_offset_list.append(position_offset)
            kf_list.append(Kf)
            joint_positions.append(robot.sim.data.qpos[robot._ref_joint_vel_indexes].copy())
            joint_velocities.append(dq.copy())
        
        if visualize and getattr(env, "has_renderer", False):
            env.render()
            time.sleep(speed)
        
        step_idx += 1
    
    elapsed = time.time() - start_time
    print(f"✓ Rollout complete in {elapsed:.2f} seconds")
    env.close()
    
    # Plot results
    if plot_results and record_data:
        plot_rollout_results(
            desired_pos_list, actual_pos_list, 
            force_z_list, force_correction_list,
            position_offset_list, kf_list,
            save=save_plot
        )
    
    # Return recorded data if requested
    if record_data:
        return {
            'desired_pos': np.array(desired_pos_list),
            'actual_pos': np.array(actual_pos_list),
            'force_z': np.array(force_z_list),
            'force_correction': np.array(force_correction_list),
            'position_offset': np.array(position_offset_list),
            'kf': np.array(kf_list),
            'joint_positions': np.array(joint_positions),
            'joint_velocities': np.array(joint_velocities),
            'elapsed_time': elapsed,
            'steps': step_idx
        }
    return None

# ============================================================================
# 8. PLOTTING FUNCTION
# ============================================================================
def plot_rollout_results(desired_pos_list, actual_pos_list, 
                         force_z_list, force_correction_list,
                         position_offset_list, kf_list,
                         save=True):
    """Generate standard performance plots from rollout data."""
    if len(actual_pos_list) == 0:
        print("No data to plot.")
        return
    
    desired_pos = np.array(desired_pos_list)
    actual_pos = np.array(actual_pos_list)
    force_z = np.array(force_z_list)
    force_corr = np.array(force_correction_list)
    pos_offset = np.array(position_offset_list)
    kf_vals = np.array(kf_list)
    time_steps = np.arange(len(actual_pos)) * 0.005
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("Policy Rollout - Dynamic Gain Control", fontsize=14, fontweight="bold")
    
    axes[0].plot(time_steps, desired_pos[:, 2], "r--", label="Desired Z", linewidth=2)
    axes[0].plot(time_steps, actual_pos[:, 2], "b-", label="Actual Z", linewidth=1.5)
    axes[0].set_ylabel("Z Position [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(time_steps, force_z, "b-", label="Measured Fz", linewidth=1.5)
    axes[1].axhline(y=1.5, color="r", linestyle="--", label="Desired Force")
    axes[1].set_ylabel("Z Force [N]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(time_steps, kf_vals, "g-", linewidth=2, label="NN Output $K_f$")
    axes[2].set_ylabel("Adaptation Gain $K_f$")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    axes[3].plot(time_steps, force_corr, "purple", label="Force Correction", linewidth=1.5)
    axes[3].set_ylabel("Force Correction [N]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    axes[4].plot(time_steps, pos_offset * 1000, "teal", label="Z Position Offset", linewidth=1.5)
    axes[4].set_xlabel("Time [s]")
    axes[4].set_ylabel("Offset [mm]")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    
    plt.tight_layout()
    
    if save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"rollout_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300)
        print(f"✓ Plot saved to: {filename}")
    
    plt.show()

# ============================================================================
# 9. MAIN - DEMONSTRATION/TRAINING
# ============================================================================
def train_and_save_policy():
    """Train a new policy using CMA-ES and save it."""
    print("\n" + "="*60)
    print("TRAINING NEW POLICY")
    print("="*60)
    
    # Create environment for training
    env = suite.make(
        env_name="Wipe",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        horizon=1000000,
        seed=SEED,
    )
    
    env.sim.model.viscontype = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
    env.sim.model.visconsize = 0.01
    env.sim.model.visconrgb = np.array([1, 0, 0])
    
    policy_nn = PolicyNN(input_dim=12, hidden_dim=16)
    num_params = policy_nn.num_params
    
    # Define cost function
    def evaluate_policy(params_vec, env, max_steps=MAX_EVAL_STEPS):
        policy_nn.set_params(params_vec)
        obs = env.reset()
        robot = env.robots[0]
        eef_id = get_eef_site_id(robot)
        eef_name = robot.sim.model.site_id2name(eef_id)
        
        force_ctrl = ForceController(
            desired_force=1.5, kp_force=80.0, ki_force=0.8,
            max_force_correction=1.0, max_integral=8.0, force_deadzone=0.3
        )
        
        total_cost = 0.0
        step_idx = 0
        limit_steps = min(num_steps, max_steps) if max_steps is not None else num_steps
        
        while step_idx < limit_steps:
            x_desired = pos_traj[step_idx].copy()
            q_desired = quat_traj[step_idx].copy()
            
            x_current = get_eef_position(robot)
            quat_current = T.mat2quat(get_eef_rotation_matrix(robot))
            dq = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()
            
            if np.dot(q_desired, quat_current) < 0:
                q_desired = -q_desired
            
            ee_wrench_world = get_eef_wrench_from_ft_sensor(
                env, robot, tool_mass=TOOL_MASS,
                tool_com_eef=TOOL_COM_EEF, add_noise=True
            )
            
            current_force_z = ee_wrench_world[2]
            force_correction = force_ctrl.update(current_force_z, dt=0.005)
            force_err = force_ctrl.force_errors[-1] if force_ctrl.force_errors else 0.0
            
            pos_err_raw = x_desired - x_current
            state_vec = np.concatenate([pos_err_raw, dq, [current_force_z], [force_err]])
            
            Kf, Kp_pos_diag, Kp_ori_diag = policy_nn.forward(state_vec)
            
            Kp_pos = np.diag(Kp_pos_diag)
            Kp_ori = np.diag(Kp_ori_diag)
            Kd_pos = 2 * np.sqrt(Kp_pos)
            Kd_ori = 2 * np.sqrt(Kp_ori)
            
            Kp = np.block([[Kp_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kp_ori]])
            Kd = np.block([[Kd_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kd_ori]])
            
            position_offset = Kf * force_correction
            x_desired_modified = x_desired.copy()
            x_desired_modified[2] += position_offset
            
            max_position_offset = 0.05
            x_desired_modified[2] = np.clip(x_desired_modified[2], 
                                            x_desired[2] - max_position_offset, 
                                            x_desired[2] + max_position_offset)
            
            pos_error = x_desired_modified - x_current
            ori_error = T.get_orientation_error(q_desired, quat_current)
            cartesian_error = np.concatenate([pos_error, ori_error])
            
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
            
            # Cost computation
            w_xy = 500.0
            w_ori = 100.0
            w_contact = 300.0
            w_overload = 20.0
            F_min = 0.5
            F_max = 2.0
            
            e_xy = pos_err_raw[:2]
            J_track = w_xy * np.sum(e_xy**2) + w_ori * np.sum(ori_error**2)
            J_contact = w_contact * max(0.0, F_min - current_force_z)**2
            J_overload = w_overload * max(0.0, current_force_z - F_max)**2
            
            step_cost = J_track + J_contact + J_overload
            total_cost += step_cost
            step_idx += 1
        
        return total_cost
    
    # CMA-ES optimization
    initial_mean = np.zeros(num_params)
    initial_sigma = 0.5
    num_generations = 5  # Reduced for speed
    population_size = 8
    
    print(f"\nInitializing CMA-ES with {num_params} parameters...")
    print(f"Generations: {num_generations}, Population: {population_size}")
    
    try:
        import cma
        es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, 
                                      {'popsize': population_size, 'seed': SEED})
        for gen in range(num_generations):
            solutions = es.ask()
            costs = []
            for idx, x in enumerate(solutions):
                print(f"  Gen {gen+1}/{num_generations} - Eval {idx+1}/{population_size}...", end="\r", flush=True)
                costs.append(evaluate_policy(x, env, max_steps=MAX_EVAL_STEPS))
            print()
            es.tell(solutions, costs)
            print(f"Gen {gen+1}/{num_generations} | Best: {min(costs):.4f} | Avg: {np.mean(costs):.4f}")
        best_params = es.result.xbest
    except ImportError:
        print("CMA library not found, using simple evolution strategy...")
        mean_theta = initial_mean.copy()
        best_params = mean_theta.copy()
        best_global_cost = float("inf")
        
        for gen in range(num_generations):
            noise = np.random.normal(0, initial_sigma, size=(population_size, num_params))
            candidates = [mean_theta + noise[i] for i in range(population_size)]
            costs = []
            for idx, cand in enumerate(candidates):
                print(f"  Gen {gen+1}/{num_generations} - Eval {idx+1}/{population_size}...", end="\r", flush=True)
                costs.append(evaluate_policy(cand, env, max_steps=MAX_EVAL_STEPS))
            print()
            
            min_idx = np.argmin(costs)
            if costs[min_idx] < best_global_cost:
                best_global_cost = costs[min_idx]
                best_params = candidates[min_idx].copy()
            
            top_k = population_size // 2
            sorted_indices = np.argsort(costs)[:top_k]
            mean_theta = np.mean([candidates[i] for i in sorted_indices], axis=0)
            
            print(f"Gen {gen+1}/{num_generations} | Best: {min(costs):.4f} | Avg: {np.mean(costs):.4f}")
    
    env.close()
    
    # Save trained policy
    policy_nn.set_params(best_params)
    filename = f"policy_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    save_policy(policy_nn, filename)
    
    return policy_nn

# ============================================================================
# 10. INTERACTIVE COMMAND INTERFACE
# ============================================================================
def interactive_menu():
    """Interactive menu for policy management and rollout."""
    print("\n" + "="*60)
    print("POLICY ROLLOUT MANAGER")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("  1. List saved policies")
        print("  2. Load policy and run rollout (with visualization)")
        print("  3. Load policy and run rollout (headless, record data)")
        print("  4. Train new policy and save")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            list_saved_policies()
        
        elif choice == '2':
            list_saved_policies()
            filename = input("Enter policy filename (or press Enter for latest): ").strip()
            if not filename:
                files = [f for f in os.listdir(POLICY_DIR) if f.endswith('.pkl')]
                if files:
                    filename = sorted(files)[-1]
                    print(f"Using latest: {filename}")
                else:
                    print("No policies found.")
                    continue
            
            try:
                policy = load_policy(filename)
                run_rollout(policy, visualize=True, record_data=True, plot_results=True)
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            list_saved_policies()
            filename = input("Enter policy filename (or press Enter for latest): ").strip()
            if not filename:
                files = [f for f in os.listdir(POLICY_DIR) if f.endswith('.pkl')]
                if files:
                    filename = sorted(files)[-1]
                    print(f"Using latest: {filename}")
                else:
                    print("No policies found.")
                    continue
            
            try:
                policy = load_policy(filename)
                data = run_rollout(policy, visualize=False, record_data=True, 
                                   plot_results=True, save_plot=True)
                if data:
                    # Save recorded data
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    np.savez(f"rollout_data_{timestamp}.npz", **data)
                    print(f"✓ Data saved to: rollout_data_{timestamp}.npz")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            print("\nStarting training...")
            policy = train_and_save_policy()
            print("\nTraining complete! Running visualization of trained policy...")
            run_rollout(policy, visualize=True, record_data=True, plot_results=True)
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

# ============================================================================
# 11. QUICK START FUNCTIONS (Importable)
# ============================================================================
def quick_load_and_run(policy_filename=None):
    """Quick function to load and run the latest or specified policy."""
    if policy_filename is None:
        files = [f for f in os.listdir(POLICY_DIR) if f.endswith('.pkl')]
        if not files:
            print("No saved policies found. Train one first.")
            return
        policy_filename = sorted(files)[-1]
    
    policy = load_policy(policy_filename)
    run_rollout(policy, visualize=True, record_data=True, plot_results=True)

def quick_train_and_run():
    """Quick function to train a new policy and run it."""
    policy = train_and_save_policy()
    run_rollout(policy, visualize=True, record_data=True, plot_results=True)

# ============================================================================
# 12. MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--train":
            # Train and run
            policy = train_and_save_policy()
            run_rollout(policy, visualize=True, record_data=True, plot_results=True)
        elif sys.argv[1] == "--load":
            filename = sys.argv[2] if len(sys.argv) > 2 else None
            quick_load_and_run(filename)
        elif sys.argv[1] == "--list":
            list_saved_policies()
        else:
            print("Usage:")
            print("  python script.py --train    # Train new policy")
            print("  python script.py --load     # Load latest policy")
            print("  python script.py --list     # List saved policies")
            print("  python script.py            # Interactive menu")
    else:
        # Interactive menu
        interactive_menu()