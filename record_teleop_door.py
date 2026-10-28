import time
import random
import numpy as np
import matplotlib.pyplot as plt


import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import VisualizationWrapper
from robosuite.devices import Keyboard
from robosuite.controllers import load_composite_controller_config
import mujoco

# Set deterministic seed
SEED = 40
np.random.seed(SEED)
random.seed(SEED)


# ------------------------------------------------------------------
# 1. Physics & Sensing Helper Functions
# ------------------------------------------------------------------
def get_eef_site_id(robot):
    """Dynamically get the correct EEF site ID."""
    if hasattr(robot, "eef_site_id"):
        if isinstance(robot.eef_site_id, dict):
            return robot.eef_site_id.get("right", list(robot.eef_site_id.values())[0])
        return robot.eef_site_id
    # Fallback lookup by site name
    site_name = robot.eef_name if hasattr(robot, "eef_name") else f"{robot.name}_right_hand"
    return robot.sim.model.site_name2id(site_name)

def get_eef_geoms(env, robot):
    """Get all geometries belonging to the end-effector."""
    sim = env.sim
    eef_site_id = get_eef_site_id(robot)
    eef_geoms = set()
    
    # Method 1: Get all geoms from the end-effector body
    eef_body_id = sim.model.site_bodyid[eef_site_id]
    for g_id in range(sim.model.ngeom):
        if sim.model.geom_bodyid[g_id] == eef_body_id:
            eef_geoms.add(g_id)
    
    # Method 2: Get gripper geometries
    if hasattr(robot, "gripper") and robot.gripper is not None:
        if hasattr(robot.gripper, "contact_geoms"):
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
    """
    Gets the raw contact wrench at the end-effector frame.
    Returns wrench in EEF frame: [Fx, Fy, Fz, Tx, Ty, Tz]
    """
    sim = env.sim
    eef_site_id = get_eef_site_id(robot)
    eef_pos = sim.data.site_xpos[eef_site_id]
    eef_rotmat = sim.data.site_xmat[eef_site_id].reshape(3, 3)
    
    # Get all geometries belonging to the EEF
    eef_geoms = get_eef_geoms(env, robot)
    
    total_force = np.zeros(3)
    total_torque = np.zeros(3)
    
    num_contacts = sim.data.ncon
    
    for i in range(num_contacts):
        contact = sim.data.contact[i]
        
        is_geom1 = contact.geom1 in eef_geoms
        is_geom2 = contact.geom2 in eef_geoms
        
        if is_geom1 or is_geom2:
            # Get contact force in world frame
            c_force = np.zeros(6)
            mujoco.mj_contactForce(sim.model._model, sim.data._data, i, c_force)
            c_frame = contact.frame.reshape(3, 3)
            force_world = c_frame.T @ c_force[:3]
            
            # Correct direction
            if is_geom2:
                force_world = -force_world
            
            # Transform force to EEF frame
            force_eef = eef_rotmat.T @ force_world
            
            # Position of contact in EEF frame
            r_eef = contact.pos - eef_pos
            r_eef_local = eef_rotmat.T @ r_eef
            
            # Torque in EEF frame
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
    sim = env.sim
    eef_site_id = get_eef_site_id(robot)
    
    eef_quat = T.mat2quat(sim.data.site_xmat[eef_site_id].reshape(3, 3))
    R_eef_to_world = T.quat2mat(eef_quat)
    
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
# 2. Main Teleoperation Recording Script
# ------------------------------------------------------------------
def main():
    controller_config = load_composite_controller_config(controller="BASIC")

    env = suite.make(
        env_name="Door",
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

    env = VisualizationWrapper(env)

    device = Keyboard(env)
    device.start_control()

    print("=" * 70)
    print("TELEOPERATION RECORDER STARTED (Wipe Task)")
    print("Robot:", robot.name)
    print("Controls (Click inside MuJoCo GUI window to focus):")
    print("  Arrow Keys : Move Horizontally (X/Y plane)")
    print("  . / ;      : Move Vertically (Up/Down)")
    print("  o / p      : Rotate Yaw")
    print("  Spacebar   : Toggle Gripper Open / Close")
    print("  Ctrl + C   : Press in terminal to Stop & Save Trajectory")
    print("=" * 70)

    log_data = []
    step_cnt = 0

    try:
        while True:
            action_dict = device.input2action()

            if action_dict is None:
                continue

            action = np.concatenate(
                [
                    action_dict["right_delta"],
                    action_dict["right_gripper"],
                ]
            )

            obs, reward, done, info = env.step(action)
            env.render()

            # ==============================================================
            # State & Wrench Calculations
            # ==============================================================
            # Direct access to robot EEF position (dynamic) or observation dict
            if "robot0_eef_pos" in obs:
                eef_position = obs["robot0_eef_pos"].copy()
            else:
                eef_position = robot._eef_xpos.copy()

            eef_orientation = obs.get("robot0_eef_quat", np.array([1., 0., 0., 0.])).copy()

            # World Contact Wrench calculation
            wrench_world = get_eef_wrench_from_ft_sensor(env, robot, add_noise=True)
            force_world = wrench_world[:3]
            torque_world = wrench_world[3:]

            if step_cnt % 20 == 0:
                print(f"Step {step_cnt:05d} | "
                      f"EEF Pos [m]: ({eef_position[0]:.3f}, {eef_position[1]:.3f}, {eef_position[2]:.3f}) | "
                      f"Force World [N]: ({force_world[0]:.2f}, {force_world[1]:.2f}, {force_world[2]:.2f}) | "
                      f"Torque World [Nm]: ({torque_world[0]:.2f}, {torque_world[1]:.2f}, {torque_world[2]:.2f})")

            joint_positions = robot.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
            joint_velocities = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()
            gripper_status = obs["robot0_gripper_qpos"].copy() if "robot0_gripper_qpos" in obs else np.array([])

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
                "wrench_world": wrench_world,
                "force_world": force_world,
                "torque_world": torque_world,
                "gripper": gripper_status,
                "action": action.copy(),
                "reward": reward,
                "joint_torques": joint_torques,
            }

            log_data.append(sample)
            step_cnt += 1
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nRecording stopped by user (Ctrl+C). Saving data...")

    if len(log_data) > 0:
        save_path = "teleop_trajectory_door.npz"

        pos_arr = np.array([s["eef_position"] for s in log_data])
        quat_arr = np.array([s["eef_orientation"] for s in log_data])
        wrench_arr = np.array([s["wrench_world"] for s in log_data])
        force_arr = np.array([s["force_world"] for s in log_data])
        torque_arr = np.array([s["torque_world"] for s in log_data])
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
            wrench=wrench_arr,
            force=force_arr,
            torque=torque_arr,
            gripper=gripper_arr,
            action=action_arr,
            qpos=qpos_arr,
            qvel=qvel_arr
        )

        print(f"Successfully saved {len(log_data)} trajectory steps to '{save_path}'!")
    else:
        print("No data recorded.")

if __name__ == "__main__":
    main()