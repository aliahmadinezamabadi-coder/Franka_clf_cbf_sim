import time
import random
import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
import mujoco
import osqp
from scipy import sparse

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
    
    # -----------------------------------------------------------
    # External Wrench in Local Tool (EEF) Frame
    # -----------------------------------------------------------
    force_eef = -wrench_compensated[:3]
    torque_eef = -wrench_compensated[3:]
    ee_wrench_tool = np.concatenate([force_eef, torque_eef])
    
    # -----------------------------------------------------------
    # External Wrench in World Frame
    # -----------------------------------------------------------
    force_world = R_eef_to_world @ force_eef
    torque_world = R_eef_to_world @ torque_eef
    ee_wrench_world = np.concatenate([force_world, torque_world])
    
    return ee_wrench_world, ee_wrench_tool

def get_joint_dynamics(robot):
    """Returns joint-space inertia matrix M and bias forces h (Coriolis + Gravity)."""
    qpos = robot.sim.data.qpos[robot._ref_joint_pos_indexes]
    qvel = robot.sim.data.qvel[robot._ref_joint_vel_indexes]
    
    M = np.zeros((robot.dof, robot.dof))
    mujoco.mj_fullM(robot.sim.model._model, M, robot.sim.data.qM)
    M = M[np.ix_(robot._ref_joint_vel_indexes, robot._ref_joint_vel_indexes)]
    
    h = robot.sim.data.qfrc_bias[robot._ref_joint_vel_indexes].copy()
    return M, h

def get_operational_space_matrices(J, M):
    """Computes Operational Space Mass Matrix Lambda and dynamically consistent pseudo-inverse J_bar."""
    M_inv = np.linalg.inv(M)
    Lambda_inv = J @ M_inv @ J.T
    
    Lambda = np.linalg.pinv(Lambda_inv, rcond=1e-4)
    J_bar = M_inv @ J.T @ Lambda  
    N = np.eye(M.shape[0]) - J.T @ J_bar.T  
    return Lambda, J_bar, N

# ------------------------------------------------------------------
# 4. Controller Definition (Exact 6D Stiffness QP Adaptation)
# ------------------------------------------------------------------
class Unified_CBF_CLF_Controller:
    """
    Unified QP solver using OSQP matching C++ 6D stiffness rate optimization:
    Decision vector: y = [u_n, u_K_pos(3), u_K_ori(3), delta]^T  (8 Variables)
    """      

    def __init__(self, 
                 desired_force=3.0,
                 max_allowed_force=5.0,
                 lambda_clf=0.5,
                 gamma_cbf_force=1.0,
                 gamma_cbf_pos=1.0,
                 gamma_k=0.5,
                 p_gains=(10.0, 0.1),
                 w_slack=1e4,
                 w_k=100.0,
                 surface_stiffness=400.0,
                 d_safe=0.003,
                 F_threshold=1.0,
                 alpha_filter=0.3):
        
        self.desired_force = desired_force
        self.max_allowed_force = max_allowed_force
        self.lambda_clf = lambda_clf
        self.gamma_cbf_force = gamma_cbf_force
        self.gamma_cbf_pos = gamma_cbf_pos
        self.gamma_k = gamma_k
        self.w_slack = w_slack
        self.w_k = w_k
        self.Ks = surface_stiffness
        self.d_safe = d_safe
        self.F_threshold = F_threshold
        self.alpha_filter = alpha_filter
        
        # P matrix gain entries p11, p12, p22 matching CPP
        self.p11 = float(p_gains[0])
        self.p12 = 0.0
        self.p22 = float(p_gains[1])
        
        self.integral_error = 0.0
        self.downward_offset = 0.0
        self.filtered_force = 0.0
        self.normal_vec_prev = np.array([0.0, 0.0, 1.0])
        self.enable_u_n_opt = True
        
        self.force_errors = []
        self.force_outputs = []

        # P Hessian Matrix for 8 variables: y = [u_n, u_K_6d(6), delta]
        self.P_mat = sparse.csc_matrix(np.diag([1.0] + [self.w_k] * 6 + [self.w_slack]))
        self.prob = osqp.OSQP()
        self.solver_initialized = False

    def update(self, 
               current_force, 
               x_current, 
               x_desired, 
               normal_vec, 
               Kp_curr_6d, 
               u_K_ref_6d, 
               K_min_6d, 
               K_max_6d, 
               surface_point=None, 
               desired_penetration=0.003, 
               dt=0.005):

        safe_dt = max(dt, 1e-4)

        # 1. Normal vector unitization & force calculation along normal direction
        n = normal_vec / (np.linalg.norm(normal_vec) + 1e-8)
        current_force_n = float(np.dot(current_force, n))
        
        # Filter Force Measurement matching C++ filter
        measured_force = self.alpha_filter * current_force_n + (1.0 - self.alpha_filter) * self.filtered_force
        self.filtered_force = measured_force
        
        # 2. Tracking Error State
        e_F = self.desired_force - measured_force
        in_contact = (measured_force > self.F_threshold)
        if in_contact:
            self.integral_error = np.clip(self.integral_error + e_F * safe_dt, -1.0, 1.0)
        else:
            self.integral_error = 0.0
        e_I = self.integral_error
        
        # 3. Exact Normal Contact Dynamic Terms (Equations 35-39 from C++ controller)
        n_sq = n**2
        Kp_pos_curr = Kp_curr_6d[:3]
        K_n = max(10.0, float(np.dot(n_sq, Kp_pos_curr)))
        K_s = self.Ks
        K_e = (K_s * K_n) / (K_s + K_n)

        s_r = (measured_force / K_e) if (K_e > 1e-4) else 0.0

        n_dot = (n - self.normal_vec_prev) / safe_dt
        self.normal_vec_prev = n.copy()

        v_s = 0.0  # Rigid surface assumption
        a_F = K_e
        b_F = (s_r * K_s * K_s / ((K_s + K_n) ** 2)) * n_sq
        f_F = K_e * v_s + 2.0 * (s_r * K_s * K_s / ((K_s + K_n) ** 2)) * float(np.dot(n_dot, Kp_pos_curr * n))

        # 4. Strict CLF Lie Derivatives matching C++ formulation
        V_F = 0.5 * (self.p11 * e_F * e_F + 2.0 * self.p12 * e_F * e_I + self.p22 * e_I * e_I)
        chi_F = self.p11 * e_F + self.p12 * e_I
        psi_F = self.p12 * e_F + self.p22 * e_I

        F_d_dot = 0.0
        Lf_VF = chi_F * (F_d_dot - f_F) + psi_F * e_F
        Lgn_VF = -chi_F * a_F
        LgK_VF = -chi_F * b_F

        # 5. Barrier Terms
        h_F = self.max_allowed_force - measured_force
        
        if surface_point is None:
            surface_point = x_desired

        distance_normal = float(np.dot(x_current - surface_point, n))
        h_P = distance_normal + self.d_safe

        # Linear cost vector: q = [q_un, -w_k * u_K_ref_6d, 0]
        q_un = -2.0 if not in_contact else 0.0  # Linear push downward in free space
        q = np.concatenate([[q_un], -self.w_k * u_K_ref_6d, [0.0]])

        # Construct Constraint Matrix A (l <= A * y <= u) for 8 variables
        # Row 0: Relaxed Force CLF
        row_clf = [Lgn_VF, LgK_VF[0], LgK_VF[1], LgK_VF[2], 0.0, 0.0, 0.0, -1.0]
        # Row 1: Upper Force CBF: -a_F * u_n - b_F . u_K_pos >= f_F - gamma_f * h_F
        row_force_cbf = [-a_F, -b_F[0], -b_F[1], -b_F[2], 0.0, 0.0, 0.0, 0.0]
        # Row 2: Surface Penetration CBF: -u_n >= -gamma_p * h_P
        row_pos_cbf = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Row 3: Slack Non-Negativity: delta >= 0
        row_slack = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # Row 4: Normal Velocity Command u_n Hard Box Bound
        row_u_n = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Rows 5-10: 6D Stiffness Rate Bounds (u_K_pos and u_K_ori)
        stiff_A = np.hstack([np.zeros((6, 1)), np.eye(6), np.zeros((6, 1))])

        A_dense = np.vstack([
            row_clf,
            row_force_cbf,
            row_pos_cbf,
            row_slack,
            row_u_n,
            stiff_A
        ])
        A = sparse.csc_matrix(A_dense)

        max_u_n = 0.05  # Bound matching C++ setup [m/s]
        l = np.array([
            -np.inf,
            f_F - self.gamma_cbf_force * h_F,
            -self.gamma_cbf_pos * h_P,
            0.0,
            -max_u_n,
            -self.gamma_k * (Kp_curr_6d[0] - K_min_6d[0]),
            -self.gamma_k * (Kp_curr_6d[1] - K_min_6d[1]),
            -self.gamma_k * (Kp_curr_6d[2] - K_min_6d[2]),
            -self.gamma_k * (Kp_curr_6d[3] - K_min_6d[3]),
            -self.gamma_k * (Kp_curr_6d[4] - K_min_6d[4]),
            -self.gamma_k * (Kp_curr_6d[5] - K_min_6d[5])
        ])

        u = np.array([
            -self.lambda_clf * V_F - Lf_VF,
            np.inf,
            np.inf,
            np.inf,
            max_u_n,
            self.gamma_k * (K_max_6d[0] - Kp_curr_6d[0]),
            self.gamma_k * (K_max_6d[1] - Kp_curr_6d[1]),
            self.gamma_k * (K_max_6d[2] - Kp_curr_6d[2]),
            self.gamma_k * (K_max_6d[3] - Kp_curr_6d[3]),
            self.gamma_k * (K_max_6d[4] - Kp_curr_6d[4]),
            self.gamma_k * (K_max_6d[5] - Kp_curr_6d[5])
        ])

        # 7. Setup & Solve OSQP Problem
        if not self.solver_initialized:
            self.prob.setup(self.P_mat, q, A, l, u, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
            self.solver_initialized = True
        else:
            self.prob.update(q=q, l=l, u=u)
            self.prob.update(Ax=A.data)

        res = self.prob.solve()

        if res.info.status == 'solved':
            u_n_opt = res.x[0]
            u_K_opt = res.x[1:7]
        else:
            u_n_opt = 0.01 if not in_contact else 0.0
            u_K_opt = np.clip(u_K_ref_6d, -self.gamma_k * (Kp_curr_6d - K_min_6d), self.gamma_k * (K_max_6d - Kp_curr_6d))

        # Ensure minimum descent speed during free-space contact search
        if not in_contact and u_n_opt < 0.005:
            u_n_opt = 0.01  # [m/s] minimum approach rate

        # Dynamic Offset Step Scaling
        max_step_free = 0.05     
        max_step_contact = 0.007 

        force_range = max(1e-3, self.desired_force - self.F_threshold)
        force_ratio = np.clip((measured_force - self.F_threshold) / force_range, 0.0, 1.0)

        max_offset_step = max_step_free - force_ratio * (max_step_free - max_step_contact)

        offset_delta = np.clip(u_n_opt * safe_dt, -max_offset_step, max_offset_step)
        
        self.downward_offset += offset_delta
        self.downward_offset = np.clip(self.downward_offset, 0.0, 1.65)

        self.force_errors.append(e_F)
        self.force_outputs.append(self.downward_offset)

        pos_offset_3d = -self.downward_offset * n

        return pos_offset_3d, u_K_opt

# ------------------------------------------------------------------
# 5. Anisotropic (Per-Axis) Adaptive Gain Parameters
# ------------------------------------------------------------------
# Translational Parameters
K_min_vec = np.array([200.0, 300.0, 300.0])  
K_max_vec = np.array([1200.0, 1200.0, 1200.0])
zeta_p_vec = np.array([0.05, 0.05, 0.05])    
eta_p_vec = np.array([6.5, 6.5, 6.5])
alpha_gain_vec = np.array([500.0, 500.0, 500.0])
lambda_decay_vec = np.array([0.2, 0.2, 0.2])
beta_gain_vec = np.array([5000.0, 5000.0, 5000.0])  

# Rotational Parameters (Matching C++ Setup)
K_ori_min_vec = np.array([500.0, 500.0, 500.0])
K_ori_max_vec = np.array([800.0, 800.0, 800.0])
zeta_o_vec = np.array([0.1, 0.1, 0.1])
eta_o_vec = np.array([2.0, 2.0, 2.0])
alpha_ori_gain_vec = np.array([50.0, 50.0, 50.0])
lambda_ori_decay_vec = np.array([0.1, 0.1, 0.1])

Kp_pos_vec = K_min_vec.copy()
Kp_ori_vec = K_ori_min_vec.copy()

k_c = 20.0                 
F_threshold = 0.2          

# ------------------------------------------------------------------
# 6. Initialize Controller & Loggers
# ------------------------------------------------------------------
force_ctrl = Unified_CBF_CLF_Controller(
    desired_force=1.0,
    max_allowed_force=8.0,
    lambda_clf=0.5,
    gamma_cbf_force=1.0,
    gamma_cbf_pos=1.0,
    gamma_k=0.5,         
    p_gains=(10.0, 0.1),
    w_slack=1e4,
    w_k=100.0             
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
force_correction_list = []
force_error_list = []
position_offset_list = []
gamma_list = []

tau_task_list = []
tau_total_list = []
tau_joints_list = []

kp_pos_list = []
kp_ori_list = []
u_K_ref_list = []  
u_K_opt_list = []  

TOOL_MASS = 0.3
TOOL_COM_EEF = np.array([0, 0, 0.02])

# ------------------------------------------------------------------
# 7. Trajectory Replay Loop
# ------------------------------------------------------------------
eef_id = get_eef_site_id(robot)
eef_name = robot.sim.model.site_id2name(eef_id)

step_idx = 0
print("\nStarting Force-Controlled Impedance Tracking Replay...")
print("="*110)
print(f"{'Step':>5} | {'EE Position (X, Y, Z) [m]':>28} | {'World Forces (Fx, Fy, Fz) [N]':>30} | {'Tool Torques (Tx, Ty, Tz) [Nm]':>30}")
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

        # ----------------------------------------------------------
        # Read External Wrench (World Frame & Tool Frame)
        # ----------------------------------------------------------
        ee_wrench_world, ee_wrench_tool = get_eef_wrench_from_ft_sensor(
            env, robot,
            tool_mass=TOOL_MASS,
            tool_com_eef=TOOL_COM_EEF,
            add_noise=True,
            force_noise_std=0.05,
            torque_noise_std=0.005
        )
        wrench_list.append(ee_wrench_world.copy())

        # Extract external force and torque directly in the Tool Frame
        force_tool = ee_wrench_tool[:3]    # [Fx, Fy, Fz] in tool frame
        torque_tool = ee_wrench_tool[3:]   # [Tx, Ty, Tz] in tool frame

        raw_sensor_wrench = get_contact_wrench_at_eef(env, robot)
        wrench_sensor_list.append(raw_sensor_wrench.copy())

        current_force_z = ee_wrench_world[2]
        force_z_list.append(current_force_z)

        if step_idx % 20 == 0:
            pos_str = f"[{x_current[0]:.3f}, {x_current[1]:.3f}, {x_current[2]:.3f}]"
            force_str = f"[{ee_wrench_world[0]:.2f}, {ee_wrench_world[1]:.2f}, {ee_wrench_world[2]:.2f}]"
            torque_str = f"[{torque_tool[0]:.3f}, {torque_tool[1]:.3f}, {torque_tool[2]:.3f}]"
            print(f"{step_idx:5d} | {pos_str:>28} | {force_str:>30} | Tool Torque (Tx, Ty, Tz): {torque_str:>25}")

        # ----------------------------------------------------------
        # 1. Contact State
        # ----------------------------------------------------------
        F_z = abs(force_ctrl.filtered_force)
        gamma = 1.0 / (1.0 + np.exp(-k_c * (F_z - F_threshold)))
        gamma_list.append(gamma)

        # ----------------------------------------------------------
        # 2. Dynamic Adaptive Stiffness Reference Rate u_K_ref
        # ----------------------------------------------------------
        dt_step = 0.005  

        e_p_vec = x_desired - x_current
        e_p_abs = np.abs(e_p_vec)

        R_s = T.quat2mat(q_desired)
        normal_vec = -R_s[:, 2]
        n = normal_vec / (np.linalg.norm(normal_vec) + 1e-8)
        n_sq = n**2  

        gamma_vec = gamma * n_sq

        scaled_error = (e_p_abs / zeta_p_vec) ** eta_p_vec
        growth_term = np.log(1.0 + scaled_error)

        F_measured = force_ctrl.filtered_force
        F_d = force_ctrl.desired_force
        force_ratio_term = (F_d - F_measured) / F_d
        desired_force_list.append(F_d)

        # Translational Reference Stiffness Rate
        u_K_pos_ref = (
            -lambda_decay_vec * (Kp_pos_vec - K_min_vec)
            + (1.0 - gamma_vec) * alpha_gain_vec * growth_term
            + gamma_vec * beta_gain_vec * force_ratio_term
        )

        # Rotational Reference Stiffness Rate (Equation matching C++)
        ori_error = T.get_orientation_error(q_desired, quat_current)
        e_o_abs = np.abs(ori_error)
        ori_scaled_error = (e_o_abs / zeta_o_vec) ** eta_o_vec
        ori_growth_term = np.log(1.0 + ori_scaled_error)

        u_K_ori_ref = (
            -lambda_ori_decay_vec * (Kp_ori_vec - K_ori_min_vec)
            + alpha_ori_gain_vec * ori_growth_term
        )

        u_K_ref_6d = np.concatenate([u_K_pos_ref, u_K_ori_ref])
        Kp_curr_6d = np.concatenate([Kp_pos_vec, Kp_ori_vec])
        K_min_6d = np.concatenate([K_min_vec, K_ori_min_vec])
        K_max_6d = np.concatenate([K_max_vec, K_ori_max_vec])

        u_K_ref_list.append(u_K_ref_6d.copy())

        ee_force_world = ee_wrench_world[:3]

        # ----------------------------------------------------------
        # 3. Unified QP Solve along End-Effector Normal
        # ----------------------------------------------------------
        pos_offset_3d, u_K_opt_6d = force_ctrl.update(
            current_force=ee_force_world,
            x_current=x_current,
            x_desired=x_desired,
            normal_vec=normal_vec,
            Kp_curr_6d=Kp_curr_6d,
            u_K_ref_6d=u_K_ref_6d,
            K_min_6d=K_min_6d,
            K_max_6d=K_max_6d,
            surface_point=x_desired,       
            desired_penetration=0.003,    
            dt=dt_step
        )

        u_K_opt_list.append(u_K_opt_6d.copy())
        force_correction_list.append(np.linalg.norm(pos_offset_3d))
        force_error_list.append(force_ctrl.force_errors[-1] if force_ctrl.force_errors else 0)
        position_offset_list.append(np.linalg.norm(pos_offset_3d))

        x_desired_modified = x_desired + pos_offset_3d

        # Rate Saturation Clamping (Matching C++ Integration Step)
        max_k_step_pos = 300.0 * dt_step
        max_k_step_ori = 50.0 * dt_step

        delta_K_pos = np.clip(u_K_opt_6d[:3] * dt_step, -max_k_step_pos, max_k_step_pos)
        delta_K_ori = np.clip(u_K_opt_6d[3:] * dt_step, -max_k_step_ori, max_k_step_ori)

        Kp_pos_vec += delta_K_pos
        Kp_ori_vec += delta_K_ori

        Kd_pos_vec = 20.0 * np.sqrt(Kp_pos_vec)
        Kd_ori_vec = 20.0 * np.sqrt(Kp_ori_vec)

        kp_pos_list.append(Kp_pos_vec.copy())
        kp_ori_list.append(Kp_ori_vec.copy())  

        Kp_pos = np.diag(Kp_pos_vec)
        Kd_pos = np.diag(Kd_pos_vec)
        Kp_ori = np.diag(Kp_ori_vec)
        Kd_ori = np.diag(Kd_ori_vec)

        Kp = np.block([[Kp_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kp_ori]])
        Kd = np.block([[Kd_pos, np.zeros((3, 3))], [np.zeros((3, 3)), Kd_ori]])

        # ----------------------------------------------------------
        # 4. Impedance Control & Actuation
        # ----------------------------------------------------------
        pos_error = x_desired_modified - x_current
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

        M_q, h_q = get_joint_dynamics(robot)
        Lambda, J_bar, N = get_operational_space_matrices(J, M_q)

        K_null = 500.0 * np.eye(robot.dof)
        tau_null = N @ (-K_null @ dq)

        F_task = Kp @ cartesian_error - Kd @ dx
        tau_task = J.T @ F_task

        tau_total = tau_task + h_q + tau_null
        tau_total = np.clip(tau_total, -max_torque, max_torque)

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
    kp_pos_arr = np.array(kp_pos_list)      
    kp_ori_arr = np.array(kp_ori_list)      
    u_K_ref_arr = np.array(u_K_ref_list)    
    u_K_opt_arr = np.array(u_K_opt_list)    
    tau_joints_arr = np.atleast_2d(np.array(tau_joints_list))  

    time_steps = np.arange(len(actual_pos)) * 0.005
    kp_ref_integrated = K_min_6d + np.cumsum(u_K_ref_arr * dt_step, axis=0)

    fig1, axes = plt.subplots(9, 1, figsize=(10, 27), sharex=True)
    
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

    axes[2].plot(time_steps, force_z, "b-", label="Measured Fz")
    axes[2].plot(time_steps, desired_force_arr, "r--", label="Target Sinusoidal Force")
    axes[2].set_ylabel("Z Force [N]")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(time_steps, force_corr, "purple", label="CBF/CLF Output [m]")
    axes[3].set_ylabel("Force Correction [m]")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="upper right", fontsize=8)

    axes[4].plot(time_steps, pos_offset * 1000, "teal", label="Blended Offset [mm]")
    axes[4].plot(time_steps, gamma * 100, "g--", label="Gamma (%)")
    axes[4].set_ylabel("Offset / Gamma")
    axes[4].grid(True, linestyle="--", alpha=0.6)
    axes[4].legend(loc="upper right", fontsize=8)

    axes[5].plot(time_steps, kp_pos_arr[:, 0], color="crimson", label="Kp_x")
    axes[5].plot(time_steps, kp_pos_arr[:, 1], color="limegreen", label="Kp_y")
    axes[5].plot(time_steps, kp_pos_arr[:, 2], color="royalblue", label="Kp_z")
    axes[5].set_ylabel("Trans. Stiffness [N/m]")
    axes[5].grid(True, linestyle="--", alpha=0.6)
    axes[5].legend(loc="upper right", ncol=3, fontsize=8)

    ori_colors = ["darkorange", "purple", "brown"]
    ori_labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    for i in range(3):
        axes[6].plot(time_steps, kp_ori_arr[:, i], color=ori_colors[i], linewidth=1.5, label=f"Kp_ori_{ori_labels[i]}")
    axes[6].set_ylabel("Ori. Stiffness [Nm/rad]")
    axes[6].grid(True, linestyle="--", alpha=0.6)
    axes[6].legend(loc="upper right", ncol=3, fontsize=8)

    colors = ["crimson", "limegreen", "royalblue"]
    axis_names = ["X", "Y", "Z"]
    for i in range(3):
        axes[7].plot(time_steps, u_K_ref_arr[:, i], "--", color=colors[i], alpha=0.6, label=f"u_K_ref_{axis_names[i]}")
        axes[7].plot(time_steps, u_K_opt_arr[:, i], "-", color=colors[i], linewidth=1.5, label=f"u_K_opt_{axis_names[i]}")
    axes[7].set_ylabel("Stiffness Rate [N/m/s]")
    axes[7].grid(True, linestyle="--", alpha=0.6)
    axes[7].legend(loc="upper right", ncol=3, fontsize=8)

    for i in range(3):
        axes[8].plot(time_steps, kp_ref_integrated[:, i], "--", color=colors[i], alpha=0.7, label=f"Integrated Kp_ref_{axis_names[i]}")
        axes[8].plot(time_steps, kp_pos_arr[:, i], "-", color=colors[i], linewidth=1.5, label=f"Actual Kp_opt_{axis_names[i]}")
    axes[8].set_xlabel("Time [s]")
    axes[8].set_ylabel("Integrated Stiffness [N/m]")
    axes[8].grid(True, linestyle="--", alpha=0.6)
    axes[8].legend(loc="upper right", ncol=3, fontsize=8)

    fig1.tight_layout()
    fig1.savefig("cbf_clf_wiping_results.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    joint_colors = ["crimson", "darkorange", "goldenrod", "limegreen", "teal", "royalblue", "purple"]

    num_joints = tau_joints_arr.shape[1] if tau_joints_arr.ndim > 1 else 0
    for j in range(num_joints):
        ax2.plot(time_steps[:len(tau_joints_arr)], tau_joints_arr[:, j], color=joint_colors[j % len(joint_colors)], linewidth=1.2, label=f"Joint {j+1}")

    ax2.axhline(y=max_torque, color="r", linestyle="--", alpha=0.6, label=f"±{max_torque} Nm Limit")
    ax2.axhline(y=-max_torque, color="r", linestyle="--", alpha=0.6)
    ax2.set_title(r"7-DoF Franka Panda Joint Control Torques ($\tau_{total}$)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Torque [Nm]")
    ax2.set_ylim(-max_torque * 1.1, max_torque * 1.1)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", ncol=4, fontsize=8)

    fig2.tight_layout()
    fig2.savefig("joint_torques_standalone.png", dpi=300)

    plt.show(block=True)