import time
import random
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config

# Set global random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def generate_constant_speed_trajectory_by_duration(waypoints, target_duration=30.0, dt=0.005):
    """
    Interpolates discrete 3D waypoints into a dense time-series trajectory designed to 
    last exactly `target_duration` seconds at a constant linear speed.
    """
    # 1. Calculate total cumulative path distance along waypoints
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    total_distance = np.sum(segment_lengths)

    # 2. Compute required speed to complete the distance in target_duration
    target_speed = total_distance / target_duration
    step_dist = target_speed * dt  # Spatial step size per 5 ms control tick

    dense_positions = [waypoints[0].copy()]
    current_pt = waypoints[0].copy()

    for i in range(len(waypoints) - 1):
        p_start = waypoints[i]
        p_end = waypoints[i + 1]

        segment_vec = p_end - p_start
        segment_len = np.linalg.norm(segment_vec)

        if segment_len < 1e-6:
            continue

        unit_vec = segment_vec / segment_len
        accumulated_dist = 0.0

        while accumulated_dist + step_dist <= segment_len:
            current_pt = current_pt + unit_vec * step_dist
            dense_positions.append(current_pt.copy())
            accumulated_dist += step_dist

        current_pt = p_end.copy()

    return np.array(dense_positions), target_speed, total_distance


# ------------------------------------------------------------------
# 1. Environment & Controller Setup
# ------------------------------------------------------------------
controller_config = load_composite_controller_config(controller="BASIC")

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
eef_id = robot.eef_site_id["right"] if isinstance(robot.eef_site_id, dict) else robot.eef_site_id

# ------------------------------------------------------------------
# 2. Target Heights and Orientation Setup
# ------------------------------------------------------------------
table_height = env.model.mujoco_arena.table_top_abs[2]
target_z = table_height - 0.009  # 9 mm surface penetration

print(f"Table Surface Z: {table_height:.4f} m | Wiping Z: {target_z:.4f} m")

# Target Orientation: End-effector pointing straight down
quat_target_down = np.array([0.0, 1.0, 0.0, 0.0])  # [w, x, y, z]

# ------------------------------------------------------------------
# 3. Generate Wiping Grid & Constant-Speed Trajectory (30 Seconds)
# ------------------------------------------------------------------
x_min, x_max = 0.1, 0.2
y_min, y_max = -0.15, 0.22

# Option: Increase num_passes (e.g., 9 passes) for a denser grid, or keep 5 passes for slower speed
num_passes = 5
grid_waypoints = []
y_sweeps = np.linspace(y_min, y_max, num_passes)

for i, y in enumerate(y_sweeps):
    x_line = np.linspace(x_min, x_max, 10)
    if i % 2 != 0:
        x_line = x_line[::-1]

    for x in x_line:
        grid_waypoints.append([x, y, target_z])

grid_waypoints = np.array(grid_waypoints)

# Target trajectory settings
TARGET_DURATION = 30.0  # seconds
DT = 0.005              # seconds per control loop

const_pos_traj, computed_speed, path_dist = generate_constant_speed_trajectory_by_duration(
    grid_waypoints, target_duration=TARGET_DURATION, dt=DT
)
num_traj_steps = len(const_pos_traj)
actual_duration = num_traj_steps * DT

print("=" * 80)
print(f"Path Length:     {path_dist:.3f} m")
print(f"Target Duration: {TARGET_DURATION:.1f} s ({int(TARGET_DURATION / DT)} steps)")
print(f"Actual Trajectory: {num_traj_steps} steps (~{actual_duration:.2f} s execution time)")
print(f"Computed Constant Speed: {computed_speed:.4f} m/s ({computed_speed * 100:.2f} cm/s)")
print("=" * 80)

# ------------------------------------------------------------------
# 4. Closed-Loop Execution Loop
# ------------------------------------------------------------------
log_data = []

print("\nExecuting 30-second constant-speed automated wiping trajectory... (Press Ctrl+C to stop)")

try:
    for step_idx in range(num_traj_steps):
        target_pos = const_pos_traj[step_idx]

        current_pos = robot.sim.data.site_xpos[eef_id].copy()
        current_mat = robot.sim.data.site_xmat[eef_id].reshape(3, 3)
        current_quat = T.mat2quat(current_mat)

        # Position tracking step
        pos_error = target_pos - current_pos

        # Orientation tracking step
        if np.dot(quat_target_down, current_quat) < 0:
            current_quat = -current_quat

        ori_error = T.get_orientation_error(quat_target_down, current_quat)

        # Build action command vector
        arm_action = np.concatenate([pos_error, ori_error])[:6]

        if env.action_dim > 6:
            action = np.concatenate([arm_action, [0.0]])
        else:
            action = arm_action

        obs, reward, done, info = env.step(action)
        env.render()

        joint_positions = robot.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
        joint_velocities = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()

        eef_position = obs["robot0_eef_pos"].copy()
        eef_orientation = obs["robot0_eef_quat"].copy()
        gripper_status = obs["robot0_gripper_qpos"].copy()

        try:
            joint_torques = robot.torques.copy()
        except Exception:
            joint_torques = None

        sim_time = env.sim.data.time

        sample = {
            "time": sim_time,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "eef_position": eef_position,
            "eef_orientation": eef_orientation,
            "gripper": gripper_status,
            "action": action.copy(),
            "reward": reward,
            "joint_torques": joint_torques,
        }

        log_data.append(sample)
        time.sleep(0.005)

    print("\nAutomated wiping task completed successfully!")

except KeyboardInterrupt:
    print("\n\nExecution stopped by user (Ctrl+C). Saving data...")

# ------------------------------------------------------------------
# 5. Save Trajectory Data
# ------------------------------------------------------------------
if len(log_data) > 0:
    save_path = "auto_wiping_trajectory.npz"

    pos_arr = np.array([s["eef_position"] for s in log_data])
    quat_arr = np.array([s["eef_orientation"] for s in log_data])
    gripper_arr = np.array([s["gripper"] for s in log_data])
    action_arr = np.array([s["action"] for s in log_data])
    qpos_arr = np.array([s["joint_positions"] for s in log_data])
    qvel_arr = np.array([s["joint_velocities"] for s in log_data])
    time_arr = np.array([s["time"] for s in log_data])

    np.savez(
        save_path,
        time=time_arr,
        pos=pos_arr,
        quat=quat_arr,
        gripper=gripper_arr,
        action=action_arr,
        qpos=qpos_arr,
        qvel=qvel_arr,
    )

    print(f"Successfully saved {len(log_data)} trajectory steps (~{time_arr[-1] - time_arr[0]:.2f}s) to '{save_path}'!")
else:
    print("No data recorded.")