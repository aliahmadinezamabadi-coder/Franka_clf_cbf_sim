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
    seed=SEED,  # Fixed seed ensures dirt pattern is deterministic
)

obs = env.reset()
robot = env.robots[0]
eef_id = robot.eef_site_id["right"]

# ------------------------------------------------------------------
# 2. Target Heights and Orientation Setup
# ------------------------------------------------------------------
table_height = env.model.mujoco_arena.table_top_abs[2]
target_z = table_height - 0.009  # Slight penetration into table to maintain contact

print(f"Table Surface Z: {table_height:.4f} m | Wiping Z: {target_z:.4f} m")

# Target Orientation: End-effector pointing straight down
quat_target_down = np.array([0.0, 1.0, 0.0, 0.0]) # [w, x, y, z]

# ------------------------------------------------------------------
# 3. Generate Wiping Grid (X, Y Waypoints in World Frame)
# ------------------------------------------------------------------
x_min, x_max = 0.1, 0.2
y_min, y_max = -0.15, 0.22

num_passes = 5
waypoints = []
y_sweeps = np.linspace(y_min, y_max, num_passes)

for i, y in enumerate(y_sweeps):
    x_line = np.linspace(x_min, x_max, 10)
    if i % 2 != 0:
        x_line = x_line[::-1]
    
    for x in x_line:
        waypoints.append([x, y, target_z])

waypoints = np.array(waypoints)
print(f"Generated {len(waypoints)} wiping waypoints.")

# ------------------------------------------------------------------
# 4. Storage & Closed-Loop Execution Loop
# ------------------------------------------------------------------
pos_gain = 0.75       
max_step_size = 0.25     
ori_gain = 0.75          
pos_threshold = 0.015    

log_data = []            

print("\nExecuting smooth, controlled automated wiping pattern... (Press Ctrl+C to stop)")

try:
    for wp_idx, target_pos in enumerate(waypoints):
        reached = False
        step_count = 0
        max_steps_per_wp = 300 

        while not reached and step_count < max_steps_per_wp:
            current_pos = robot.sim.data.site_xpos[eef_id].copy()
            current_mat = robot.sim.data.site_xmat[eef_id].reshape(3, 3)
            current_quat = T.mat2quat(current_mat)

            pos_error = target_pos - current_pos
            distance = np.linalg.norm(pos_error)

            if distance < pos_threshold:
                reached = True
                break

            delta_pos_action = np.clip(
                pos_gain * pos_error, 
                -max_step_size, 
                max_step_size
            )

            if np.dot(quat_target_down, current_quat) < 0:
                current_quat = -current_quat
            
            ori_error = T.get_orientation_error(quat_target_down, current_quat)
            delta_ori_action = np.clip(ori_gain * ori_error, -1.0, 1.0)

            arm_action = np.concatenate([delta_pos_action, delta_ori_action])[:6]
            
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
            step_count += 1

    print("Automated wiping task completed successfully!")

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
        qvel=qvel_arr           
    )

    print(f"Successfully saved {len(log_data)} trajectory steps to '{save_path}'!")
else:
    print("No data recorded.")