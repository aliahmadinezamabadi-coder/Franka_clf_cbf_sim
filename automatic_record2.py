import robosuite as suite
from robosuite.devices import Keyboard
from robosuite.controllers import load_composite_controller_config
import numpy as np

def main():
    # 1. Set up environment with appropriate controller
    controller_config = load_composite_controller_config(controller="OSC_POSE")
    
    env = suite.make(
        env_name="Door",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        use_camera_obs=False,
        control_freq=20,
    )

    # 2. Initialize Keyboard device
    device = Keyboard(env=env)
    
    # Reset environment
    obs = env.reset()
    device.start_control()

    print("\n" + "=" * 60)
    print("Teleoperation Started!")
    print("Controls:")
    print("  W/S - Move forward/backward")
    print("  A/D - Move left/right") 
    print("  R/F - Move up/down")
    print("  Z/X - Rotate about X axis")
    print("  C/V - Rotate about Y axis")
    print("  B/N - Rotate about Z axis")
    print("  Spacebar - Toggle gripper")
    print("  ESC - Quit")
    print(f"Action dimension: {env.action_dim}")
    print("=" * 60 + "\n")

    # Manual action handling
    # The Keyboard device in robosuite returns a 3D action for position
    # We need to convert it to 7D for the robot
    def process_action(raw_action):
        if raw_action is None:
            return np.zeros(env.action_dim)
        
        # Keyboard returns a 3D action (x, y, z) when using default settings
        # We need to expand it to 7D (x, y, z, roll, pitch, yaw, gripper)
        if len(raw_action) == 3:
            # Position control only
            full_action = np.zeros(7)
            full_action[:3] = raw_action  # x, y, z
            full_action[3:6] = [0, 0, 0]  # No rotation
            full_action[6] = 0  # Gripper open
            return full_action
        elif len(raw_action) == 6:
            # Position + orientation control
            full_action = np.zeros(7)
            full_action[:6] = raw_action
            full_action[6] = 0  # Gripper open
            return full_action
        else:
            # Already the correct size
            return raw_action[:env.action_dim]

    try:
        step_count = 0
        while True:
            # Get raw action from keyboard
            raw_action = device.input2action()
            
            # Process to correct dimension
            action = process_action(raw_action)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            env.render()
            
            step_count += 1
            
            # Print status every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count} - Action: {action[:3]}...")
            
            # Check for ESC key to exit (handled by device.get_controller_state)
            state = device.get_controller_state()
            if state["reset"]:
                break
                
    except KeyboardInterrupt:
        print("\nTeleoperation interrupted by user.")
    except Exception as e:
        print(f"Error during teleoperation: {e}")
    finally:
        env.close()
        print(f"\nTeleoperation ended. Total steps: {step_count}")

if __name__ == "__main__":
    main()