import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


print('running... waiting for connection with Unity simulation')
env = UnityEnvironment(file_name=None)

try:
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    
    print("Connected! Running... Press Ctrl+C to stop.")
    while True:
        env_out = env.step()  # Ticks the physics loop forward 1 frame

except KeyboardInterrupt:
    pass

finally:
    env.close()  # Unfreezes Unity gracefully
    print("Disconnected safely.")