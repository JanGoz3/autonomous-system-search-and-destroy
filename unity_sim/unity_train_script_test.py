import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.base_env import ActionTuple

print('running... waiting for connection with Unity simulation')
env = UnityEnvironment(file_name=None)

try:
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]   

    print("Connected! Running... Press Ctrl+C to stop.")
    while True:
        # ask unity for the current state of the agents
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(decision_steps) > 0:
            
            sensor_vector = decision_steps.obs[0]
            print(sensor_vector)

            num_agents = len(decision_steps)

            with torch.no_grad():
                action_tensor = torch.rand((num_agents, 2)) * 2 - 1

            action_numpy = action_tensor.cpu().numpy()

            action_tuple = ActionTuple(continuous=action_numpy)

            env.set_actions(behavior_name, action_tuple)

        env.step()  # Ticks the physics loop forward 1 frame
except KeyboardInterrupt:
    print('stopped by user')
except Exception as e:
    print(f'an error occured: {e}')

finally:
    env.close()  # Unfreezes Unity gracefully
    print("Disconnected safely.")