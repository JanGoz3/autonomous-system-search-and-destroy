import pickle
from collections import defaultdict

import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

OUTPUT_PATH = "dt_dataset.pkl"
MAX_EPISODES = 50
SAVE_EVERY_N_EPISODES = 5

print("running... waiting for connection with Unity simulation")
env = UnityEnvironment(file_name=None)

current_traj = defaultdict(lambda: {"obs": [], "actions": [], "rewards": []})

finished_episodes = []


def finalize_episode(agent_id):

    traj = current_traj[agent_id]
    if len(traj["rewards"]) == 0:
        return

    rewards = np.array(traj["rewards"], dtype=np.float32)
    rtg = np.cumsum(rewards[::-1])[::-1].copy()

    episode = {
        "observations": np.array(traj["obs"], dtype=np.float32),
        "actions": np.array(traj["actions"], dtype=np.float32),
        "rewards": rewards,
        "returns_to_go": rtg,
    }
    finished_episodes.append(episode)
    current_traj[agent_id] = {"obs": [], "actions": [], "rewards": []}

    print(
        f"[epizod {len(finished_episodes)}] dlugosc={len(rewards)} "
        f"suma_nagrod={rewards.sum():.3f}"
    )


def save_dataset(path):
    with open(path, "wb") as f:
        pickle.dump(finished_episodes, f)
    print(f"Zapisano {len(finished_episodes)} epizodow do: {path}")


try:
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    print("Connected! Zbieranie danych... Ctrl+C aby przerwac wczesniej.")

    while len(finished_episodes) < MAX_EPISODES:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        for idx, agent_id in enumerate(terminal_steps.agent_id):
            final_reward = terminal_steps.reward[idx]

            if len(current_traj[agent_id]["rewards"]) > 0:
                current_traj[agent_id]["rewards"][-1] += final_reward
            finalize_episode(agent_id)

            if len(finished_episodes) % SAVE_EVERY_N_EPISODES == 0:
                save_dataset(OUTPUT_PATH)

        if len(decision_steps) > 0:
            obs_batch = decision_steps.obs[0]
            num_agents = len(decision_steps)

            with torch.no_grad():
                action_tensor = torch.rand((num_agents, 2)) * 2 - 1

            action_numpy = action_tensor.cpu().numpy()
            action_tuple = ActionTuple(continuous=action_numpy)
            env.set_actions(behavior_name, action_tuple)

            for i, agent_id in enumerate(decision_steps.agent_id):
                current_traj[agent_id]["obs"].append(obs_batch[i])
                current_traj[agent_id]["actions"].append(action_numpy[i])
                current_traj[agent_id]["rewards"].append(decision_steps.reward[i])

        env.step()

except KeyboardInterrupt:
    print("stopped by user")
except Exception as e:
    print(f"an error occured: {e}")
finally:
    save_dataset(OUTPUT_PATH)
    env.close()
    print("Disconnected safely.")