import time
import random
import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
import mujoco
from scipy.optimize import minimize

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
    # --- ADD Z OFFSET HERE ---
    z_offset = 0.08  # Shift upward by 0.05 meters (5 cm)
    pos_traj[:, 2] += z_offset
    # -------------------------

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
    
    force_world = R_eef_to_world @ force_eef
    torque_world = R_eef_to_world @ torque_eef
    
    return np.concatenate([force_world, torque_world])


def get_joint_dynamics(robot):
    """Returns joint-space inertia matrix M and bias forces h (Coriolis + Gravity)."""
    qpos = robot.sim.data.qpos[robot._ref_joint_pos_indexes]
    qvel = robot.sim.data.qvel[robot._ref_joint_vel_indexes]
    
    # Joint mass matrix M (7x7)
    M = np.zeros((robot.dof, robot.dof))
    mujoco.mj_fullM(robot.sim.model._model, M, robot.sim.data.qM)
    M = M[np.ix_(robot._ref_joint_vel_indexes, robot._ref_joint_vel_indexes)]
    
    # Bias forces h = C(q, dq)*dq + g(q)
    h = robot.sim.data.qfrc_bias[robot._ref_joint_vel_indexes].copy()
    return M, h

def get_operational_space_matrices(J, M):
    """
    Computes Operational Space Mass Matrix Lambda and dynamically consistent pseudo-inverse J_bar.
    Lambda = (J * M^-1 * J^T)^-1
    """
    M_inv = np.linalg.inv(M)
    Lambda_inv = J @ M_inv @ J.T
    
    # Regularization for stability near singularities
    Lambda = np.linalg.pinv(Lambda_inv, rcond=1e-4)
    J_bar = M_inv @ J.T @ Lambda  # Dynamically consistent pseudo-inverse
    N = np.eye(M.shape[0]) - J.T @ J_bar.T  # Null-space projection operator
    return Lambda, J_bar, N

# ------------------------------------------------------------------
# 4. Controller Definition
# ------------------------------------------------------------------
class Unified_CBF_CLF_Controller:
    """
    Unified QP solver optimizing displacement rate u_z, stiffness derivatives u_K = d(Kp)/dt, and slack delta.
    Decision vector: y = [u_z, u_K_x, u_K_y, u_K_z, delta]^T
    """
    def __init__(self, 
                 desired_force=0.90,
                 max_allowed_force=1.0,
                 lambda_clf=10.0,
                 gamma_cbf_force=15.0,
                 gamma_cbf_pos=100.0,
                 gamma_k=5.0,
                 p_gains=(10.0, 0.1),
                 w_slack=1e4,
                 w_k=0.1,
                 surface_stiffness=400.0):
        
        self.desired_force = desired_force
        self.max_allowed_force = max_allowed_force
        self.lambda_clf = lambda_clf
        self.gamma_cbf_force = gamma_cbf_force
        self.gamma_cbf_pos = gamma_cbf_pos
        self.gamma_k = gamma_k
        self.w_slack = w_slack
        self.w_k = w_k
        self.Ks = surface_stiffness
        
        self.P = np.diag([p_gains[0], p_gains[1]])
        self.integral_error = 0.0
        self.downward_offset = 0.0
        self.filtered_force = 0.0
        self.alpha = 0.3
        
        self.force_errors = []
        self.force_outputs = []

    def update(self, 
               current_force, 
               x_current, 
               x_desired, 
               normal_vec, 
               Kp_curr,          
               u_K_ref,          
               K_min_vec,        
               K_max_vec,        
               surface_point=None, 
               desired_penetration=0.003, 
               desired_force=None,   # <-- ADD THIS PARAMETER
               dt=0.005):        
        
        # Override desired force if passed dynamically
        if desired_force is not None:
            self.desired_force = desired_force

        n = normal_vec / (np.linalg.norm(normal_vec) + 1e-8)
        
        # ... rest of update method remains the same ...

        # Force measurement along normal direction (scalar)
        current_force_n = float(np.dot(current_force, n))
        
        # Filter Force Measurement
        self.filtered_force = self.alpha * abs(current_force_n) + (1 - self.alpha) * self.filtered_force
        measured_force = self.filtered_force
        
        # 2. Tracking Error State x = [e_F, integral_e_F]^T
        e_F = self.desired_force - measured_force
        self.integral_error = np.clip(self.integral_error + e_F * dt, -1.0, 1.0)
        x_state = np.array([e_F, self.integral_error])
        
        # Position tracking error along normal direction
        e_n = float(np.dot(x_desired - x_current, n))
        
        #  Note: Kp_curr[:3] corresponds to translational stiffness [Kp_x, Kp_y, Kp_z]
        K_n = float(np.dot(n**2, Kp_curr[:3]))
        
        # 3. Lie Derivatives for Force CLF
        V = 0.5 * float(x_state.T @ self.P @ x_state)
        LfV = float(self.P[1, 1] * x_state[1] * x_state[0])
        
        # Coefficient for normal velocity u_n in CLF: LgV_un = -p1 * (K_s + K_n) * e_F
        LgV_un = -self.P[0, 0] * (self.Ks + K_n) * e_F
        
        # 6D Coefficient vector for u_K in CLF (zero-padded for orientation stiffness components)
        # Translational components act on e_n * n_i^2; rotational components do not affect normal force directly
        LgV_uK = np.zeros(6)
        LgV_uK[:3] = self.P[0, 0] * e_F * e_n * (n**2)

        # 4. Force & Position CBF terms
        h_force = self.max_allowed_force - measured_force
        
        # Default surface point to x_desired if not explicitly passed
        if surface_point is None:
            surface_point = x_desired

        # Signed distance to local surface along normal vector
        distance_normal = float(np.dot(x_current - surface_point, n))

        # Bounded penetration barrier: h_pos >= 0 means tool stays above penetration depth limit
        h_pos = distance_normal + desired_penetration

        # 5. Objective Function (Decision Vector y = [u_n, u_K_pos(3), u_K_ori(3), delta] -> size 8)
        def objective(y):
            u_n = y[0]
            u_K = y[1:7]
            delta = y[7]
            return 0.5 * (u_n**2) + 0.5 * self.w_k * np.sum((u_K - u_K_ref)**2) + 0.5 * self.w_slack * (delta**2)

        def grad_objective(y):
            u_n = y[0]
            u_K = y[1:7]
            delta = y[7]
            return np.concatenate([[u_n], self.w_k * (u_K - u_K_ref), [self.w_slack * delta]])

        # 6. QP Constraints
        # Decision Vector: y = [u_n, u_K_x, u_K_y, u_K_z, delta]
        constraints = [
            # CLF Constraint: LfV + LgV_un * u_n + LgV_uK . u_K - delta <= -lambda * V
            {'type': 'ineq', 'fun': lambda y: (-self.lambda_clf * V - LfV) - (LgV_un * y[0] + np.dot(LgV_uK, y[1:7]) - y[7])},
            
            # Force CBF: (Ks + Kn) * u_n - e_n * (n^2 . u_K) + gamma_f * h_force >= 0
            {'type': 'ineq', 'fun': lambda y: (self.Ks + K_n) * y[0] - e_n * np.dot(n**2, y[1:4]) + self.gamma_cbf_force * h_force},
            
            # Position CBF along surface normal: h_p = n^T(x-x_s)+d_p, dh_p/dt = -u_n, -u_n + gamma_p*h_pos >= 0
            {'type': 'ineq', 'fun': lambda y: -y[0] + self.gamma_cbf_pos * h_pos},
            
            # Slack non-negativity: delta >= 0
            {'type': 'ineq', 'fun': lambda y: y[7]}
        ]

        # Stiffness Bounds CBFs
        for i in range(6):
            constraints.append({
                'type': 'ineq', 
                'fun': (lambda i=i: lambda y: -y[1 + i] + self.gamma_k * (K_max_vec[i] - Kp_curr[i]))()
            })
            constraints.append({
                'type': 'ineq', 
                'fun': (lambda i=i: lambda y: y[1 + i] + self.gamma_k * (Kp_curr[i] - K_min_vec[i]))()
            })

        # Initial Guess y0
        y0 = np.concatenate([[0.05], u_K_ref, [0.0]])
        res = minimize(objective, y0, jac=grad_objective, constraints=constraints, method='SLSQP')
        
        if res.success:
            u_n_opt = res.x[0]
            u_K_opt = res.x[1:7]
        else:
            u_n_opt = max(0.0, -LgV_un)
            u_K_opt = np.clip(u_K_ref, -self.gamma_k * (Kp_curr - K_min_vec), self.gamma_k * (K_max_vec - Kp_curr))

        # Integrate displacement offset along normal direction
        self.downward_offset += u_n_opt * dt
        self.downward_offset = np.clip(self.downward_offset, 0.0, 0.15)

        self.force_errors.append(e_F)
        self.force_outputs.append(self.downward_offset)

        # Project scalar offset along 3D normal direction vector
        pos_offset_3d = -self.downward_offset * n

        return pos_offset_3d, u_K_opt

# ------------------------------------------------------------------
# 5. Anisotropic (Per-Axis) Adaptive Gain Parameters
# ------------------------------------------------------------------
# Minimum stiffness per axis [x, y, z] in N/m
K_min_vec = np.array([200.0, 300.0, 300.0])  

# Maximum upper bound cap per axis in N/m
K_max_vec = np.array([2000.0, 2000.0, 2000.0])

# Allowable error boundaries per axis [m] (e.g., 3cm in X/Y, 5cm in Z)
zeta_p_vec = np.array([0.01, 0.01, 0.05])    

# Growth exponent per axis
eta_p_vec = np.array([6.5, 6.5, 6.5])

# Growth rate coefficient alpha_i per axis [N / (m * s)]
alpha_gain_vec = np.array([500.0, 500.0, 500.0])

# Restoration / decay rate lambda_i per axis [1/s]
lambda_decay_vec = np.array([0.2, 0.2, 0.2])

beta_gain_vec = np.array([5000.0, 5000.0, 5000.0])  # Adjust per-axis weights as needed

# Initial adaptive stiffness state vector
Kp_pos_vec = K_min_vec.copy()


# --- Replace fixed orientation gains ---
# Minimum & Maximum orientation stiffness per axis [roll, pitch, yaw] [Nm/rad]
K_ori_min_vec = np.array([200.0, 200.0, 200.0])
K_ori_max_vec = np.array([1200.0, 1200.0, 1200.0])

# Adaptation parameters for orientation
zeta_o_vec = np.array([0.05, 0.05, 0.05])       # Allowable orientation error boundary [rad]
eta_o_vec = np.array([6.5, 6.5, 6.5])           # Growth exponent
alpha_ori_gain_vec = np.array([300.0, 300.0, 300.0])  # Growth rate [Nm/(rad*s)]
lambda_ori_decay_vec = np.array([0.2, 0.2, 0.2])     # Restoration rate [1/s]

# Initial orientation stiffness state
Kp_ori_vec = K_ori_min_vec.copy()







eef_id = get_eef_site_id(robot)
eef_name = robot.sim.model.site_id2name(eef_id)

# Contact Blending Parameters
k_c = 20.0                 # Sharpness gain for contact transition
F_threshold = 0.2          # Force threshold [N] for contact detection

# ------------------------------------------------------------------
# 6. Initialize Controller & Loggers
# ------------------------------------------------------------------
# Sinusoidal Force Profile Parameters: F_d(t) = F_mean + F_amp * sin(2 * pi * f * t)
F_mean = 1.0     # Mean force [N]
F_amp = 0.5      # Sine amplitude [N] (Force fluctuates between 0.5 N and 1.5 N)
freq = 0.5       # Frequency [Hz]

force_ctrl = Unified_CBF_CLF_Controller(
    desired_force=F_mean,
    max_allowed_force=2.0,  # Ensure max allowed force is higher than peak sine (1.5 N)
    lambda_clf=10.0,
    gamma_cbf_force=15.0,
    gamma_cbf_pos=100.0,
    gamma_k=5.0,
    p_gains=(1.0, 1.1),
    w_slack=1e4,
    w_k=0.01
)

desired_force_list = []  # Add a list to log time-varying target forces

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
gamma_list = []

tau_task_list = []
tau_total_list = []
tau_joints_list = []


kp_pos_list = []
kp_ori_list = []
u_K_ref_list = []  # NEW: Store reference stiffness rates
u_K_opt_list = []  # NEW: Store optimized stiffness rates

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
        # 5. Corrected 
            force_noise_std=0.05,
            torque_noise_std=0.005
        )
        wrench_list.append(ee_wrench_world.copy())

        raw_sensor_wrench = get_contact_wrench_at_eef(env, robot)
        wrench_sensor_list.append(raw_sensor_wrench.copy())

        current_force_z = ee_wrench_world[2]
        force_z_list.append(current_force_z)

        if step_idx % 20 == 0:
            pos_str = f"[{x_current[0]:.3f}, {x_current[1]:.3f}, {x_current[2]:.3f}]"
            force_str = f"[{ee_wrench_world[0]:.2f}, {ee_wrench_world[1]:.2f}, {ee_wrench_world[2]:.2f}]"
            torque_str = f"[{ee_wrench_world[3]:.3f}, {ee_wrench_world[4]:.3f}, {ee_wrench_world[5]:.3f}]"
            print(f"{step_idx:5d} | {pos_str:>28} | {force_str:>30} | {torque_str:>30}")

        # ----------------------------------------------------------
        # 1. Contact State
        # ----------------------------------------------------------
        F_z = abs(force_ctrl.filtered_force)

        # Compute continuous contact state gamma(f_z)
        gamma = 1.0 / (1.0 + np.exp(-k_c * (F_z - F_threshold)))
        gamma_list.append(gamma)

        # ----------------------------------------------------------
        # 2. Dynamic Adaptive Stiffness Reference Rate u_K_ref
        # ----------------------------------------------------------
        dt_step = 0.005  # Integration time step [s]

        # Element-wise Cartesian position error vector using nominal trajectory
        e_p_vec = x_desired - x_current
        e_p_abs = np.abs(e_p_vec)


        # Extract orientation error vector (roll, pitch, yaw errors)
        ori_error = T.get_orientation_error(q_desired, quat_current)
        ori_error_abs = np.abs(ori_error)


        # 1. Compute surface normal vector and directional weights
        R_s = T.quat2mat(q_desired)
        normal_vec = -R_s[:, 2]
        n = normal_vec / (np.linalg.norm(normal_vec) + 1e-8)
        n_sq = n**2  # Directional projection weights (n_x^2, n_y^2, n_z^2)

        # 2. Project scalar contact state gamma across Cartesian axes based on normal orientation
        # n_sq smoothly distributes gamma to X, Y, Z based on tilt
        gamma_vec = gamma * n_sq

        # 3. Position error term (Free-space tracking)
        e_p_vec = x_desired - x_current
        e_p_abs = np.abs(e_p_vec)
        scaled_error = (e_p_abs / zeta_p_vec) ** eta_p_vec
        growth_term = np.log(1.0 + scaled_error)


        # 1. Dynamic Orientation Stiffness Rate Reference (u_K_ori_ref)
        ori_scaled_error = (ori_error_abs / zeta_o_vec) ** eta_o_vec
        ori_growth_term = np.log(1.0 + ori_scaled_error)


        # Compute elapsed time
        t_sim = step_idx * dt_step

        # Calculate dynamic sinusoidal reference force
        F_d = F_mean + F_amp * np.sin(2.0 * np.pi * freq * t_sim)
        desired_force_list.append(F_d)

        # Update force ratio term with time-varying F_d
        F_measured = force_ctrl.filtered_force
        force_ratio_term = (F_d - F_measured) / (F_d + 1e-5)  # Safe division    


        # 5. Corrected u_K_ref calculation
        # - Free-space growth is active tangentially: (1.0 - gamma_vec)
        # - Force adaptation scales with normal projection: gamma_vec * force_ratio_term
        u_K_ref = (
            -lambda_decay_vec * (Kp_pos_vec - K_min_vec)
            + (1.0 - gamma_vec) * alpha_gain_vec * growth_term
            + gamma_vec * beta_gain_vec * force_ratio_term
        )


        u_K_ori_ref = -lambda_ori_decay_vec * (Kp_ori_vec - K_ori_min_vec) + alpha_ori_gain_vec * ori_growth_term
        # Store reference stiffness rates
        u_K_ref_list.append(u_K_ref.copy())



        # Extract end-effector orientation matrix and normal vector
        R_s = T.quat2mat(q_desired)
        normal_vec = -R_s[:, 2]  # Z-axis of end-effector frame pointing toward surface

        # Extract 3D measured force from sensor wrench
        ee_force_world = ee_wrench_world[:3]


        # Concatenate 6D stiffness references & bounds
        u_K_ref_6d = np.concatenate([u_K_ref, u_K_ori_ref])
        Kp_curr_6d = np.concatenate([Kp_pos_vec, Kp_ori_vec])
        K_min_6d = np.concatenate([K_min_vec, K_ori_min_vec])
        K_max_6d = np.concatenate([K_max_vec, K_ori_max_vec])


        # ----------------------------------------------------------
        # 3. Unified QP Solve along End-Effector Normal
        # ----------------------------------------------------------

        

        # ... (u_K_ref calculation using updated force_ratio_term) ...

        # Call controller update with time-varying desired force
        pos_offset_3d, u_K_opt_6d = force_ctrl.update(
            current_force=ee_force_world,
            x_current=x_current,
            x_desired=x_desired,
            normal_vec=normal_vec,
            Kp_curr=Kp_curr_6d,
            u_K_ref=u_K_ref_6d,
            K_min_vec=K_min_6d,
            K_max_vec=K_max_6d,
            surface_point=x_desired,
            desired_penetration=0.003,
            desired_force=F_d,     # <-- PASS DYNAMIC FORCE HERE
            dt=dt_step
        )

        
        force_correction_list.append(np.linalg.norm(pos_offset_3d))
        force_error_list.append(force_ctrl.force_errors[-1] if force_ctrl.force_errors else 0)
        position_offset_list.append(np.linalg.norm(pos_offset_3d))

        # Apply 3D normal offset to target position vector
        x_desired_modified = x_desired + pos_offset_3d

        # Split optimized 6D stiffness rates back into Position & Orientation
        u_K_pos_opt = u_K_opt_6d[:3]
        u_K_ori_opt = u_K_opt_6d[3:]

        # 3. Integrate Stiffness & Compute Damping
        Kp_pos_vec += u_K_pos_opt * dt_step
        Kd_pos_vec = 2.0 * np.sqrt(Kp_pos_vec)

        Kp_ori_vec += u_K_ori_opt * dt_step
        Kd_ori_vec = 2.0 * np.sqrt(Kp_ori_vec)



        # Append to loggers inside the main step loop:
        kp_pos_list.append(Kp_pos_vec.copy())
    
        kp_ori_list.append(Kp_ori_vec.copy())  # Log 3D vector [K_roll, K_pitch, K_yaw]
        u_K_opt_list.append(u_K_pos_opt.copy())
        
        
        # Construct 6x6 Gain Matrices
        Kp_pos = np.diag(Kp_pos_vec)
        Kd_pos = np.diag(Kd_pos_vec)
        Kp_ori = np.diag(Kp_ori_vec)
        Kd_ori = np.diag(Kd_ori_vec)

        Kp = np.block([[Kp_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kp_ori]])
        Kd = np.block([[Kd_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kd_ori]])


        # ----------------------------------------------------------
        # 3. Impedance Control & Actuation
        # ----------------------------------------------------------
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

        # 1. Fetch joint dynamics M(q) and h(q, dq)
        M_q, h_q = get_joint_dynamics(robot)
        
        # 2. Operational-Space Inertia Matrix Lambda(q) and Null-space matrix N(q)
        Lambda, J_bar, N = get_operational_space_matrices(J, M_q)

        # Null-space joint velocity damping to stabilize internal degrees of freedom
        K_null = 5.0 * np.eye(robot.dof)
        tau_null = N @ (-K_null @ dq)

        F_task = Kp @ cartesian_error - Kd @ dx
        tau_task = J.T @ F_task
        tau_gravity = robot.sim.data.qfrc_bias[robot._ref_joint_vel_indexes].copy()

        tau_total = tau_task + h_q + tau_null

        tau_task_list.append(np.linalg.norm(tau_task))
        tau_total_list.append(np.linalg.norm(tau_total))
        tau_joints_list.append(tau_total.copy())

        normalized_action = np.clip(tau_total / max_torque, -1.0, 1.0)

        action = np.concatenate([normalized_action, [-1.0]]) if env.action_dim > 7 else normalized_action

        obs, reward, done, info = env.step(action)
        env.render()

        time.sleep(0.005)
        step_idx += 1

    print("="*110)
    print("Trajectory playback finished successfully!")

except KeyboardInterrupt:
    print("\nReplay stopped early by user.")

# ------------------------------------------------------------------
# 8. Plotting Summary
# ------------------------------------------------------------------
# Safely close the environment and destroy OpenGL contexts first to prevent GLX collisions
try:
    env.close()
    if hasattr(env, "viewer") and env.viewer is not None:
        env.viewer = None
except Exception:
    pass

if len(actual_pos_list) > 0:
    print("\nGenerating plots...")

    desired_pos = np.array(desired_pos_list)
    actual_pos = np.array(actual_pos_list)
    desired_quat = np.array(desired_quat_list)
    actual_quat = np.array(actual_quat_list)
    force_z = np.array(force_z_list)
    desired_force_arr = np.array(desired_force_list)
    force_corr = np.array(force_correction_list)
    pos_offset = np.array(position_offset_list)
    gamma = np.array(gamma_list)
    kp_pos_arr = np.array(kp_pos_list)      # (N, 3) -> [Kp_x, Kp_y, Kp_z]
    kp_ori_arr = np.array(kp_ori_list)      # (N, 3) -> [Kp_roll, Kp_pitch, Kp_yaw]
    u_K_ref_arr = np.array(u_K_ref_list)    # (N, 3)
    u_K_opt_arr = np.array(u_K_opt_list)    # (N, 3)
    tau_joints_arr = np.atleast_2d(np.array(tau_joints_list))  # (N, 7)

    time_steps = np.arange(len(actual_pos)) * 0.005

    # ------------------------------------------------------------------
    # 9-Subplot Grid for Task-Space Metrics & Joint Torques
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(9, 1, figsize=(10, 26), sharex=True)

    # Subplot 1: EE Position
    pos_colors, pos_labels = ["crimson", "limegreen", "royalblue"], ["X", "Y", "Z"]
    for i in range(3):
        axes[0].plot(time_steps, desired_pos[:, i], "--", color=pos_colors[i], alpha=0.7, label=f"Desired {pos_labels[i]}")
        axes[0].plot(time_steps, actual_pos[:, i], "-", color=pos_colors[i], linewidth=1.5, label=f"Actual {pos_labels[i]}")
    axes[0].set_ylabel("Position [m]")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    # Subplot 2: Orientation Quaternion
    quat_colors, quat_labels = ["darkorange", "crimson", "limegreen", "royalblue"], ["qw", "qx", "qy", "qz"]
    for i in range(4):
        axes[1].plot(time_steps, desired_quat[:, i], "--", color=quat_colors[i], alpha=0.7, label=f"Desired {quat_labels[i]}")
        axes[1].plot(time_steps, actual_quat[:, i], "-", color=quat_colors[i], linewidth=1.5, label=f"Actual {quat_labels[i]}")
    axes[1].set_ylabel("Quaternion")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)

    # Subplot 3: Z Force Tracking
    axes[2].plot(time_steps, force_z, "b-", label="Measured Fz")
    axes[2].plot(time_steps, desired_force_arr, "r--", label="Target Sinusoidal Force")
    axes[2].set_ylabel("Z Force [N]")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right", fontsize=8)

    # Subplot 4: CBF/CLF Correction
    axes[3].plot(time_steps, force_corr, "purple", label="CBF/CLF Output [m]")
    axes[3].set_ylabel("Force Correction [m]")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="upper right", fontsize=8)

    # Subplot 5: Blended Offset & Gamma
    axes[4].plot(time_steps, pos_offset * 1000, "teal", label="Blended Offset [mm]")
    axes[4].plot(time_steps, gamma * 100, "g--", label="Gamma (%)")
    axes[4].set_ylabel("Offset / Gamma")
    axes[4].grid(True, linestyle="--", alpha=0.6)
    axes[4].legend(loc="upper right", fontsize=8)

    # Subplot 6: Translational Stiffness Gains (Kp_pos)
    axes[5].plot(time_steps, kp_pos_arr[:, 0], color="crimson", label="Kp_x")
    axes[5].plot(time_steps, kp_pos_arr[:, 1], color="limegreen", label="Kp_y")
    axes[5].plot(time_steps, kp_pos_arr[:, 2], color="royalblue", label="Kp_z")
    axes[5].set_ylabel("Trans. Stiffness [N/m]")
    axes[5].grid(True, linestyle="--", alpha=0.6)
    axes[5].legend(loc="upper right", ncol=3, fontsize=8)

    # Subplot 7: Orientation Stiffness Gains (Kp_ori)
    ori_colors = ["darkorange", "purple", "brown"]
    ori_labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    for i in range(3):
        axes[6].plot(time_steps, kp_ori_arr[:, i], color=ori_colors[i], linewidth=1.5, label=f"Kp_ori_{ori_labels[i]}")
    axes[6].set_ylabel("Ori. Stiffness [Nm/rad]")
    axes[6].grid(True, linestyle="--", alpha=0.6)
    axes[6].legend(loc="upper right", ncol=3, fontsize=8)

    # Subplot 8: Stiffness Rates (u_K)
    colors = ["crimson", "limegreen", "royalblue"]
    axis_names = ["X", "Y", "Z"]
    for i in range(3):
        axes[7].plot(time_steps, u_K_ref_arr[:, i], "--", color=colors[i], alpha=0.6, label=f"u_K_ref_{axis_names[i]}")
        axes[7].plot(time_steps, u_K_opt_arr[:, i], "-", color=colors[i], linewidth=1.5, label=f"u_K_opt_{axis_names[i]}")
    axes[7].set_ylabel("Stiffness Rate [N/m/s]")
    axes[7].grid(True, linestyle="--", alpha=0.6)
    axes[7].legend(loc="upper right", ncol=3, fontsize=8)

    # Subplot 9: Joint Control Torques (7-DoF Panda)
    joint_colors = ["crimson", "darkorange", "goldenrod", "limegreen", "teal", "royalblue", "purple"]
    num_joints = tau_joints_arr.shape[1] if tau_joints_arr.ndim > 1 else 0
    for j in range(num_joints):
        axes[8].plot(time_steps[:len(tau_joints_arr)], tau_joints_arr[:, j], color=joint_colors[j % len(joint_colors)], linewidth=1.2, label=f"Joint {j+1}")
    axes[8].axhline(y=max_torque, color="r", linestyle="--", alpha=0.6, label=f"±{max_torque} Nm Limit")
    axes[8].axhline(y=-max_torque, color="r", linestyle="--", alpha=0.6)
    axes[8].set_xlabel("Time [s]")
    axes[8].set_ylabel(r"Torque ($\tau_{total}$) [Nm]")
    axes[8].set_ylim(-max_torque * 1.1, max_torque * 1.1)
    axes[8].grid(True, linestyle="--", alpha=0.6)
    axes[8].legend(loc="upper right", ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig("cbf_clf_wiping_results.png", dpi=300)
    plt.show()