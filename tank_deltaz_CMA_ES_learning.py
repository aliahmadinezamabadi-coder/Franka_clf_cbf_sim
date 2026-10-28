import time
import random
import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
import mujoco
import imageio
import os

# Optional import for CMA-ES.
try:
    import cma
    HAS_CMA_LIB = True
except ImportError:
    HAS_CMA_LIB = False

# Set identical global random seed to replicate dirt positions
SEED = 40
np.random.seed(SEED)
random.seed(SEED)

# Limits rollout horizon during CMA-ES training to keep speed fast
MAX_EVAL_STEPS = 8000

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
    has_renderer=False,  # Headless during CMA-ES evaluation for maximum speed
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
# 4. Force Controller Class
# ------------------------------------------------------------------
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


        # ------------------------------------------------------------------
    # 3.5 Hybrid Admittance & Physical Inversion Mapping
    # ------------------------------------------------------------------
def compute_hybrid_position_offset(
    current_force_z, pos_error_z, force_correction, Kf, Kp_z, 
    f_thresh=0.1, k_c=15.0): #f_thresh Based on Sensor Noise Floor. 
    #f_thresh should be set just above the peak-to-peak force sensor noise floor in free air.
    """
    Computes position offset dynamically switching between Free-Space 
    Admittance and Constrained Contact Matrix Inversion.
    """
    # 1. Continuous contact activation factor gamma in [0, 1]
    gamma = 1.0 / (1.0 + np.exp(-k_c * (current_force_z - f_thresh)))
        
    # 2. Free-Space Offset (Standard PI Admittance scaling)
    delta_xz_free = Kf * force_correction
        
    # 3. Constrained Contact Offset (Matrix Inversion Mapping)
    inv_Kf = 1.0 / (Kf + 1e-5)
    denom = inv_Kf - Kp_z
        
    # Singularity protection near parameter matching (inv_Kf == Kp_z)
    if abs(denom) < 1e-3:
        denom = np.sign(denom) * 1e-3 if denom != 0 else 1e-3
            
    delta_xz_contact = (1.0 / denom) * Kp_z * pos_error_z
        
    # 4. Unified Offsetting Law (Sigmoidal Blending)
    delta_xz = (1.0 - gamma) * delta_xz_free + gamma * delta_xz_contact
        
    return delta_xz

# ------------------------------------------------------------------
# 5. Neural Network Policy for Dynamic Kf and Kp Adaptation
# ------------------------------------------------------------------
class PolicyNN:
    """
    Feedforward Neural Network outputting time-varying Kf and Kp gains.
    Input state: [pos_error (3), dq (7), current_force_z (1), force_error (1)] -> Dim 12
    Outputs:
      - Kf: Force feedforward gain scalar (scaled to [0.1, 5.0])
      - Kp_pos_diag: (3,) diagonal terms for Kp_pos (scaled to [1000, 10000])
      - Kp_ori_diag: (3,) diagonal terms for Kp_ori (scaled to [200, 2000])
    """
    def __init__(self, input_dim=12, hidden_dim=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 7
        
        self.w1_shape = (input_dim, hidden_dim)
        self.b1_shape = (hidden_dim,)
        self.w2_shape = (hidden_dim, self.output_dim)
        self.b2_shape = (self.output_dim,)
        
        self.num_params = (input_dim * hidden_dim) + hidden_dim + (hidden_dim * self.output_dim) + self.output_dim
        
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


####################### First forward path ###########################################################

    # def forward(self, state_vec):
    #     """
    #     Passes state through network, applies normalization, dynamic 
    #     state variation scaling, and bounds output gains.
    #     """
    #     # ---------------------------------------------------------------
    #     # 1. State Input Normalization
    #     # Maps disparate physical features into a roughly normalized range [-1, 1]
    #     # ---------------------------------------------------------------
    #     pos_err_norm = state_vec[:3] / 0.05       # ~5 cm max scale
    #     dq_norm = state_vec[3:10] / 1.0            # ~1 rad/s max scale
    #     force_norm = (state_vec[10:11] - 1.5) / 3.0 # Centered around desired force ~1.5N
    #     force_err_norm = state_vec[11:12] / 2.0    # ~2N max error scale
        
    #     normalized_state = np.concatenate([pos_err_norm, dq_norm, force_norm, force_err_norm])

    #     # ---------------------------------------------------------------
    #     # 2. Forward Pass (Without Hard Output Clipping)
    #     # ---------------------------------------------------------------
    #     h = np.tanh(normalized_state @ self.W1 + self.b1)
    #     raw_out = h @ self.W2 + self.b2
        
    #     # Smooth Sigmoid (allows true continuous gradients across full spectrum)
    #     sig_out = 1.0 / (1.0 + np.exp(-raw_out))

    #     # ---------------------------------------------------------------
    #     # 3. Dynamic State-Dependent Modulation
    #     # Modulates gains smoothly based on position error magnitude
    #     # ---------------------------------------------------------------
    #     pos_err_magnitude = np.linalg.norm(state_vec[:3])
    #     # Smoothly transitions between 0.85 (low error) and 1.15 (high error)
    #     variation_factor = 0.85 + 0.30 * np.tanh(pos_err_magnitude * 20.0)

    #     # Apply base scaling + variation factor
    #     Kf = (4.1 + sig_out[0] * 4.9) * variation_factor
    #     Kp_pos_diag = (1000.0 + sig_out[1:4] * 9000.0) * variation_factor
    #     Kp_ori_diag = (200.0 + sig_out[4:7] * 1800.0) * variation_factor

    #     # ---------------------------------------------------------------
    #     # 4. Final Safety Clipping
    #     # ---------------------------------------------------------------
    #     Kf = np.clip(Kf, 4.1, 9.0)
    #     Kp_pos_diag = np.clip(Kp_pos_diag, 1000.0, 10000.0)
    #     Kp_ori_diag = np.clip(Kp_ori_diag, 200.0, 2000.0)

    #     return Kf, Kp_pos_diag, Kp_ori_diag


#########################################################################################################


####################### Second forward path ###########################################################

    def forward(self, state_vec):
    # 1. State Input Normalization
        pos_err_norm = state_vec[:3] / 0.05       # ~5 cm max scale
        dq_norm = state_vec[3:10] / 1.0            # ~1 rad/s max scale
        force_norm = (state_vec[10:11] - 1.5) / 3.0 # Centered around ~1.5N
        force_err_norm = state_vec[11:12] / 2.0    # ~2N max error scale
        
        normalized_state = np.concatenate([pos_err_norm, dq_norm, force_norm, force_err_norm])

        # 2. Forward Pass
        h = np.tanh(normalized_state @ self.W1 + self.b1)
        raw_out = h @ self.W2 + self.b2
        sig_out = 1.0 / (1.0 + np.exp(-raw_out)) # Range [0.0, 1.0]

        # 3. Dynamic Modulation (Modulate sig_out slightly by state error)
        pos_err_magnitude = np.linalg.norm(state_vec[:3])
        # Factor centered at 1.0, shifting output by up to +/- 15%
        var_factor = 0.85 + 0.30 * np.tanh(pos_err_magnitude * 20.0)
        
        # Apply modulated sig_out clamped to [0, 1] so gains never exceed absolute boundaries
        sig_modulated = np.clip(sig_out * var_factor, 0.0, 1.0)

        # 4. Final Gain Computation (Guaranteed to stay strictly in safe ranges without flat-line clipping)
        Kf = 0.005 + sig_modulated[0] * 0.2                      # [0.01, 0.065]
        Kp_pos_diag = 400.0 + sig_modulated[1:4] * 3000.0           # [1000, 10000]
        Kp_ori_diag = 400.0 + sig_modulated[4:7] * 1500.0            # [200, 2000]

        return Kf, Kp_pos_diag, Kp_ori_diag


#########################################################################################################


####################### Third forward path ###########################################################

    # def forward(self, state_vec):
    #     """Passes state through network and scales output gains."""
    #     h = np.tanh(state_vec @ self.W1 + self.b1)
    #     raw_out = h @ self.W2 + self.b2
        
    #     sig_out = 1.0 / (1.0 + np.exp(-np.clip(raw_out, -1.5, 1.5)))
        
    #     Kf = 4.1 + sig_out[0] * 4.9                             # [4.1, 9.0]
    #     Kp_pos_diag = 1000.0 + sig_out[1:4] * 9000.0           # [1000, 10000]
    #     Kp_ori_diag = 200.0 + sig_out[4:7] * 1800.0            # [200, 2000]
        
    #     return Kf, Kp_pos_diag, Kp_ori_diag

#########################################################################################################
# Initialize Policy Network with 12 input dimensions for 7-DOF Panda arm
policy_nn = PolicyNN(input_dim=12, hidden_dim=16)

# ------------------------------------------------------------------
# 6. Trajectory Rollout & Cost Evaluation Function
# ------------------------------------------------------------------
TOOL_MASS = 0.3
TOOL_COM_EEF = np.array([0, 0, 0.02])




def evaluate_policy(params_vec, env, render=False, max_steps=MAX_EVAL_STEPS):
    """
    Evaluates policy parameter vector in simulation and returns cumulative trajectory cost.
    """
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

    # Tracking previous gain values for J_smooth penalty
    prev_Kp_pos = None
    prev_Kp_ori = None
    prev_Kf = None

    # --- ADDITION 1: Tank State Initialization ---
    E_tank = 5.0

    while step_idx < limit_steps:
        x_desired = pos_traj[step_idx].copy()
        q_desired = quat_traj[step_idx].copy()

        x_current = get_eef_position(robot)
        quat_current = T.mat2quat(get_eef_rotation_matrix(robot))
        dq = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()

        if np.dot(q_desired, quat_current) < 0:
            q_desired = -q_desired

        ee_wrench_world = get_eef_wrench_from_ft_sensor(
            env, robot,
            tool_mass=TOOL_MASS,
            tool_com_eef=TOOL_COM_EEF,
            add_noise=True,
            force_noise_std=0.05,
            torque_noise_std=0.005
        )
        
        current_force_z = ee_wrench_world[2]
        force_correction = force_ctrl.update(current_force_z, dt=0.005)
        force_err = force_ctrl.force_errors[-1] if force_ctrl.force_errors else 0.0

        # Construct NN Input State Vector (Dim 12: 3 + 7 + 1 + 1)
        pos_err_raw = x_desired - x_current
        state_vec = np.concatenate([pos_err_raw, dq, [current_force_z], [force_err]])

        # Generate Time-Varying Kf and Kp from Policy NN
        Kf, Kp_pos_diag, Kp_ori_diag = policy_nn.forward(state_vec)

        # Print Policy Outputs to Terminal (sampled every 100 steps to prevent output spam)

        # Print Policy Outputs to Terminal (sampled every 100 steps to prevent output spam)
        if step_idx % 100 == 0:
            print(
                f"[Step {step_idx:4d}] "
                f"Fz: {current_force_z:7.3f} N | "
                f"E_tank: {E_tank:6.3f} J | "
                f"Kf: {Kf:.4f} | "
                f"Kp_pos: {np.round(Kp_pos_diag, 2)}"
            )


        # Assemble Full Impedance Matrices
        Kp_pos = np.diag(Kp_pos_diag)
        Kp_ori = np.diag(Kp_ori_diag)
        Kd_pos = 2 * np.sqrt(Kp_pos)
        Kd_ori = 2 * np.sqrt(Kp_ori)

        Kp = np.block([[Kp_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kp_ori]])
        Kd = np.block([[Kd_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kd_ori]])

        # --- Unified Offsetting Law (Free-Space + Constrained Inversion) ---
        position_offset = compute_hybrid_position_offset(
            current_force_z=current_force_z,
            pos_error_z=pos_err_raw[2],
            force_correction=force_correction,
            Kf=Kf,
            Kp_z=Kp_pos_diag[2],
            f_thresh=0.1,  # Force threshold to signal contact (N)
            k_c=15.0       # Sigmoid sharpness factor
        )
        
        x_desired_modified = x_desired.copy()
        x_desired_modified[2] += position_offset
        
        max_position_offset = 0.1
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
        
        # Replace: if render: env.render()
        if render and getattr(env, "has_renderer", False):
            env.render()
            time.sleep(0.005)

        # --- ADDITION 2: Energy Tank Update Step ---
        delta_Kp = Kp_pos_diag - (prev_Kp_pos if prev_Kp_pos is not None else Kp_pos_diag)
        dE_mod = 0.5 * np.sum((pos_err_raw**2) * delta_Kp)
        dE_diss = np.sum(2.0 * np.sqrt(Kp_pos_diag) * (dx[:3]**2)) * 0.005
        E_tank = np.clip(E_tank - dE_mod + dE_diss, 0.0, 10.0)

# ==================== COST FUNCTION WEIGHTS & TERMS ====================
        w_xy = 500.0
        w_ori = 100.0

        # Contact force safety weights
        w_contact = 300.0
        w_overload = 20.0

        # Stiffness regularization
        w_K = 0.0001

        # --- ADDITION 3: Passivity Weights ---
        w_tank = 100.0
        w_rate = 0.05
        E_min = 1.0 # Minimum energy tank threshold for passivity

        # Force limits
        F_min = 0.5      # Minimum acceptable contact force (N)
        F_max = 2.0      # Maximum acceptable contact force (N)


        # 1. Pose tracking (XY position + orientation)
        # Z is excluded because it is used for force interaction
        e_xy = pos_err_raw[:2]

        J_track = (
            w_xy * np.sum(e_xy**2)
            +
            w_ori * np.sum(ori_error**2)
        )


        # 2. Contact maintenance
        # Penalize losing contact with the surface
        J_contact = (
            w_contact *
            max(0.0, F_min - current_force_z)**2
        )


        # 3. Excessive force penalty
        # Penalize pressing too hard on the surface
        J_overload = (
            w_overload *
            max(0.0, current_force_z - F_max)**2
        )

        # --- ADDITION 4: Passivity & Rate Costs ---
        J_passivity = w_tank * max(0.0, E_min - E_tank)**2
        J_rate = w_rate * np.sum(np.maximum(0.0, delta_Kp)**2)


        # 4. Stiffness regularization
        # Prevent excessively stiff impedance behavior
        J_K = (
            w_K *
            (
                np.sum(Kp_pos_diag**2)
                +
                np.sum(Kp_ori_diag**2)
            )
        )


        # Total cost
        step_cost = (
            J_track
            +
            J_contact
            +
            J_overload
            +
            J_passivity
            +
            J_rate
            # +
            # J_K
        )

        total_cost += step_cost

        # --- ADDITION 5: Save State for Next Step ---
        prev_Kp_pos = Kp_pos_diag.copy()

        step_idx += 1

    return total_cost

# ------------------------------------------------------------------
# 7. CMA-ES Optimization Loop
# ------------------------------------------------------------------
num_params = policy_nn.num_params
initial_mean = np.zeros(num_params)
initial_sigma = 0.5
num_generations = 10
population_size = 20

print(f"\nInitializing CMA-ES for Neural Network with {num_params} parameters...")

if HAS_CMA_LIB:
    es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, {'popsize': population_size, 'seed': SEED})
    for gen in range(num_generations):
        solutions = es.ask()
        costs = []
        for idx, x in enumerate(solutions):
            print(f"  --> Evaluating candidate {idx+1}/{population_size} for Generation {gen+1}...", end="\r", flush=True)
            costs.append(evaluate_policy(x, env, render=False, max_steps=MAX_EVAL_STEPS))
        print()
        es.tell(solutions, costs)
        print(f"Generation {gen+1}/{num_generations} | Best Cost: {min(costs):.4f} | Avg Cost: {np.mean(costs):.4f}")
    best_params = es.result.xbest
else:
    print("Running lightweight Evolution Strategy optimizer...")
    mean_theta = initial_mean.copy()
    best_params = mean_theta.copy()
    best_global_cost = float("inf")

    for gen in range(num_generations):
        noise = np.random.normal(0, initial_sigma, size=(population_size, num_params))
        candidates = [mean_theta + noise[i] for i in range(population_size)]
        costs = []
        for idx, cand in enumerate(candidates):
            print(f"  --> Evaluating candidate {idx+1}/{population_size} for Generation {gen+1}...", end="\r", flush=True)
            costs.append(evaluate_policy(cand, env, render=False, max_steps=MAX_EVAL_STEPS))
        print()
        
        min_idx = np.argmin(costs)
        if costs[min_idx] < best_global_cost:
            best_global_cost = costs[min_idx]
            best_params = candidates[min_idx].copy()

        top_k = population_size // 2
        sorted_indices = np.argsort(costs)[:top_k]
        mean_theta = np.mean([candidates[i] for i in sorted_indices], axis=0)

        print(f"Generation {gen+1}/{num_generations} | Best Cost: {min(costs):.4f} | Avg Cost: {np.mean(costs):.4f}")

print("\nCMA-ES Optimization Complete! Replaying best policy visualization on FULL trajectory...")

# ------------------------------------------------------------------
# 8. Replay Best Policy with Visual Rendering & Plotting
# ------------------------------------------------------------------

env.close()
env = suite.make(
    env_name="Wipe",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    ignore_done=True,
    horizon=1000000,
    seed=SEED,
)

# Run full final trajectory rollout with optimized policy
policy_nn.set_params(best_params)
obs = env.reset()
robot = env.robots[0]
eef_id = get_eef_site_id(robot)
eef_name = robot.sim.model.site_id2name(eef_id)

force_ctrl = ForceController(
    desired_force=1.5, kp_force=80.0, ki_force=0.8,
    max_force_correction=1.0, max_integral=8.0, force_deadzone=0.3
)

desired_pos_list, actual_pos_list = [], []
force_z_list, force_correction_list, position_offset_list = [], [], []
kf_list = []

# ADD THESE TWO LISTS:
kp_pos_list = []  # Stores [Kp_x, Kp_y, Kp_z]
kp_ori_list = []  # Stores [Kp_roll, Kp_pitch, Kp_yaw]

# Frame storage for video
video_frames = []
camera_name = "frontview"  # Options: "frontview", "birdview", "agentview", "sideview"

print("\nRunning final rollout and rendering frames to video...")

step_idx = 0
while step_idx < num_steps:
    x_desired = pos_traj[step_idx].copy()
    q_desired = quat_traj[step_idx].copy()

    x_current = get_eef_position(robot)
    quat_current = T.mat2quat(get_eef_rotation_matrix(robot))
    dq = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()

    if np.dot(q_desired, quat_current) < 0:
        q_desired = -q_desired

    ee_wrench_world = get_eef_wrench_from_ft_sensor(
        env, robot, tool_mass=TOOL_MASS, tool_com_eef=TOOL_COM_EEF, add_noise=True
    )

    current_force_z = ee_wrench_world[2]
    force_z_list.append(current_force_z)

    force_correction = force_ctrl.update(current_force_z, dt=0.005)
    force_correction_list.append(force_correction)
    force_err = force_ctrl.force_errors[-1] if force_ctrl.force_errors else 0.0

    state_vec = np.concatenate([x_desired - x_current, dq, [current_force_z], [force_err]])
    Kf, Kp_pos_diag, Kp_ori_diag = policy_nn.forward(state_vec)
    kf_list.append(Kf)

    # ADD THESE TWO LINES:
    kp_pos_list.append(Kp_pos_diag.copy())
    kp_ori_list.append(Kp_ori_diag.copy())

    # Print Final Policy Rollout Actions to Terminal
    if step_idx % 100 == 0:
        print(f"[FINAL ROLLOUT Step {step_idx:4d}] Kf: {Kf:.4f} | Kp_pos: {np.round(Kp_pos_diag, 2)}")

    Kp_pos = np.diag(Kp_pos_diag)
    Kp_ori = np.diag(Kp_ori_diag)
    Kd_pos = 2 * np.sqrt(Kp_pos)
    Kd_ori = 2 * np.sqrt(Kp_ori)

    Kp = np.block([[Kp_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kp_ori]])
    Kd = np.block([[Kd_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kd_ori]])

    # --- Unified Offsetting Law (Free-Space + Constrained Inversion) ---
    position_offset = compute_hybrid_position_offset(
        current_force_z=current_force_z,
        pos_error_z=(x_desired[2] - x_current[2]),
        force_correction=force_correction,
        Kf=Kf,
        Kp_z=Kp_pos_diag[2],
        f_thresh=0.1,
        k_c=15.0
    )
    position_offset_list.append(position_offset)

    x_desired_modified = x_desired.copy()
    x_desired_modified[2] += position_offset
    
    max_position_offset = 0.1
    x_desired_modified[2] = np.clip(x_desired_modified[2], 
                                    x_desired[2] - max_position_offset, 
                                    x_desired[2] + max_position_offset)

    pos_error = x_desired_modified - x_current
    ori_error = T.get_orientation_error(q_desired, quat_current)
    cartesian_error = np.concatenate([pos_error, ori_error])

    desired_pos_list.append(x_desired.copy())
    actual_pos_list.append(x_current.copy())

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

    # 2. Capture frame for video recording (sample every 2 steps for 100 FPS playback or adapt as needed)
    if step_idx % 2 == 0:
        frame = env.sim.render(
            height=720,
            width=1280,
            camera_name=camera_name
        )
        # MuJoCo renders upside down by default, so flip vertically
        frame = np.flipud(frame)
        video_frames.append(frame)

    step_idx += 1

# 3. Export video file
video_filename = "final_wiping_rollout.mp4"
print(f"\nSaving simulation video to '{video_filename}'...")
imageio.mimsave(video_filename, video_frames, fps=100)
print("Video saved successfully!")

# ------------------------------------------------------------------
# 9. Plotting Results
# ------------------------------------------------------------------


# Create an output directory if it doesn't already exist
output_dir = "simulation_plots"
os.makedirs(output_dir, exist_ok=True)

if len(actual_pos_list) > 0:
    print(f"\nGenerating and saving performance plots to '{output_dir}/'...")
    desired_pos = np.array(desired_pos_list)
    actual_pos = np.array(actual_pos_list)
    force_z = np.array(force_z_list)
    kf_vals = np.array(kf_list)
    kp_pos_vals = np.array(kp_pos_list)  # Shape: (N, 3)
    kp_ori_vals = np.array(kp_ori_list)  # Shape: (N, 3)
    
    time_steps = np.arange(len(actual_pos)) * 0.005

    # Figure 1: Tracking Performance (X, Y, Z) and Contact Force
    fig1, axes1 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig1.suptitle("Trajectory Tracking Performance & Force Profile", fontsize=14, fontweight="bold")

    # 1. X Tracking Performance
    axes1[0].plot(time_steps, desired_pos[:, 0], "r--", label="Desired X")
    axes1[0].plot(time_steps, actual_pos[:, 0], "b-", label="Actual X")
    axes1[0].set_ylabel("X Position [m]")
    axes1[0].grid(True)
    axes1[0].legend()

    # 2. Y Tracking Performance
    axes1[1].plot(time_steps, desired_pos[:, 1], "r--", label="Desired Y")
    axes1[1].plot(time_steps, actual_pos[:, 1], "g-", label="Actual Y")
    axes1[1].set_ylabel("Y Position [m]")
    axes1[1].grid(True)
    axes1[1].legend()

    # 3. Z Tracking Performance
    axes1[2].plot(time_steps, desired_pos[:, 2], "r--", label="Desired Z")
    axes1[2].plot(time_steps, actual_pos[:, 2], "purple", label="Actual Z")
    axes1[2].set_ylabel("Z Position [m]")
    axes1[2].grid(True)
    axes1[2].legend()

    # 4. Measured Normal Force (Z)
    axes1[3].plot(time_steps, force_z, "b-", label="Measured Fz (World)")
    axes1[3].axhline(y=1.5, color="r", linestyle="--", label="Desired Force (1.5N)")
    axes1[3].set_xlabel("Time [s]")
    axes1[3].set_ylabel("Z Force [N]")
    axes1[3].grid(True)
    axes1[3].legend()

    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "tracking_performance_xy.png"), dpi=300)

    # Figure 2: Neural Network Dynamic Gains (Kp and Kf)
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig2.suptitle("Policy Neural Network Dynamic Gains Output ($K_p, K_f$)", fontsize=14, fontweight="bold")

    # 1. Translational Stiffness Gains (Kp Position: X, Y, Z)
    axes2[0].plot(time_steps, kp_pos_vals[:, 0], label="$K_{p,x}$")
    axes2[0].plot(time_steps, kp_pos_vals[:, 1], label="$K_{p,y}$")
    axes2[0].plot(time_steps, kp_pos_vals[:, 2], label="$K_{p,z}$")
    axes2[0].set_ylabel("Position Gains $K_p$")
    axes2[0].grid(True)
    axes2[0].legend()

    # 2. Rotational Stiffness Gains (Kp Orientation)
    axes2[1].plot(time_steps, kp_ori_vals[:, 0], label="$K_{p,roll}$")
    axes2[1].plot(time_steps, kp_ori_vals[:, 1], label="$K_{p,pitch}$")
    axes2[1].plot(time_steps, kp_ori_vals[:, 2], label="$K_{p,yaw}$")
    axes2[1].set_ylabel("Orientation Gains $K_p$")
    axes2[1].grid(True)
    axes2[1].legend()

    # 3. Force Feedforward Gain (Kf)
    axes2[2].plot(time_steps, kf_vals, "g-", linewidth=2, label="Force Gain $K_f$")
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("Gain $K_f$")
    axes2[2].grid(True)
    axes2[2].legend()

    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "dynamic_gains_kp_kf.png"), dpi=300)
    
    # Show plots on screen and free RAM memory afterward
    plt.show()
    plt.close('all')



    