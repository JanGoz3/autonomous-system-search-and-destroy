import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from models.driverModel.driver_network import DriverNet
from models.driverModel.driver_rollout_buffer import RolloutBuffer
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import deque
import numpy as np
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import onnx
from onnx import helper, TensorProto

class TrainingTracker:
    def __init__(self, window_size = 100):
        self.episode_rewards = deque(maxlen=window_size)

    def add_reward(self, reward):
        self.episode_rewards.append(reward)

    def get_stats(self):
        if len(self.episode_rewards) == 0:
            return 0.0, 0.0
        return np.mean(self.episode_rewards), np.std(self.episode_rewards)

print('running... waiting for connection with Unity simulation')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
GAE_LAMBDA = 0.95
STATE_SPACE = 8
ACTION_SPACE = 2
LEARNING_RATE = 3e-4
PPO_EPOCHS = 4
MINIBATCH_SIZE = 512
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

engine_channel = EngineConfigurationChannel()
engine_channel.set_configuration_parameters(time_scale=5.0)
env = UnityEnvironment(file_name=None, side_channels=[engine_channel])
model = DriverNet(in_features=STATE_SPACE, out_features=ACTION_SPACE).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
tracker = TrainingTracker(window_size=100)

try:
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]   

    print("Connected! Running... Press Ctrl+C to stop.")

    decision_steps, terminal_steps = env.get_steps(behavior_name)
    nr_of_agents = len(decision_steps)
    current_episode_rewards = np.zeros(nr_of_agents)
    total_env_steps = 0
    
    # This assigns each Unity agent a permanent "row index" (0 to 9) in our PyTorch arrays. 
    id_to_idx = {agent_id: i for i, agent_id in enumerate(decision_steps.agent_id)}

    buffer = RolloutBuffer(nr_of_agents=nr_of_agents, device=device)

    # TODO: optionally add retrieving from the environment action space and state space

    while True:
        # ask unity for the current state of the agents
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(decision_steps) > 0:

            # this block here is to ensure that we always build the same dimensional array of states coming from agents
            # regardless of which ones have crashed and which ones are still playing
            state_tensor = torch.zeros((nr_of_agents, STATE_SPACE), dtype=torch.float32).to(device)
            for i, agent_id in enumerate(decision_steps.agent_id):
                if agent_id in id_to_idx:
                    state_tensor[id_to_idx[agent_id]] = torch.tensor(decision_steps.obs[0][i], dtype=torch.float32).to(device)
            for i, agent_id in enumerate(terminal_steps.agent_id):
                if agent_id in id_to_idx:
                    state_tensor[id_to_idx[agent_id]] = torch.tensor(terminal_steps.obs[0][i], dtype=torch.float32).to(device)

            # sensor vector
            # [targetx, targety, targetz, agentx, agenty, agentz, agent_velocityX, agent_velocityZ]

            # here the agent is just playing, not learning yet
            with torch.no_grad():
                action_tensor, log_prob, entropy, value = model.get_action_and_value(state_tensor)
                value = value.flatten()
                log_prob = log_prob.flatten()

            # unity will throw error if we send action to a dead agent so we filter those out
            active_actions = torch.zeros((len(decision_steps), 2))
            for i, agent_id in enumerate(decision_steps.agent_id):
                if agent_id in id_to_idx:
                    active_actions[i] = action_tensor[id_to_idx[agent_id]]

            action_numpy = active_actions.cpu().numpy()
            action_tuple = ActionTuple(continuous=action_numpy)
            env.set_actions(behavior_name, action_tuple)
        
        env.step()  # Ticks the physics loop forward 1 frame

        # get new outcomes from the step we just took
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        current_rewards = torch.zeros(nr_of_agents).to(device)
        current_dones = torch.zeros(nr_of_agents).to(device)

        # match rewards to agents that are still playing
        for i, agent_id in enumerate(decision_steps.agent_id):
            if agent_id in id_to_idx:
                idx = id_to_idx[agent_id]
                reward = float(decision_steps.reward[i])
                current_rewards[idx] = reward
                current_dones[idx] = 0.0

                current_episode_rewards[idx] += reward

        # match rewards to agents that crashed/finished
        for i, agent_id in enumerate(terminal_steps.agent_id):
            if agent_id in id_to_idx:
                idx = id_to_idx[agent_id]
                reward = float(terminal_steps.reward[i])
                current_rewards[idx] = reward
                current_dones[idx] = 1.0

                current_episode_rewards[idx] += reward
                tracker.add_reward(current_episode_rewards[idx])
                current_episode_rewards[idx] = 0.0

        buffer.insert(state_tensor, action_tensor, log_prob, value, current_rewards, current_dones)

        total_env_steps += nr_of_agents

        if buffer.step_counter == buffer.buffer_size:
            mean_rew, std_rew = tracker.get_stats()
            print(f"==================================================")
            print(f"Total Agent Steps: {total_env_steps}")
            print(f"Mean Reward (Last 100 episodes): {mean_rew:.2f}")
            print(f"Std Reward  (Last 100 episodes): {std_rew:.2f}")
            print(f"==================================================")
            print("buffer full. Training.")

            with torch.no_grad():
                next_state_tensor = torch.zeros((nr_of_agents, STATE_SPACE), dtype=torch.float32).to(device)
                for i, agent_id in enumerate(decision_steps.agent_id):
                    if agent_id in id_to_idx:
                        next_state_tensor[id_to_idx[agent_id]] = torch.tensor(decision_steps.obs[0][i], dtype=torch.float32).to(device)
                for i, agent_id in enumerate(terminal_steps.agent_id):
                    if agent_id in id_to_idx:
                        next_state_tensor[id_to_idx[agent_id]] = torch.tensor(terminal_steps.obs[0][i], dtype=torch.float32).to(device)

                next_value = model.get_value(next_state_tensor).squeeze()

            advantages = torch.zeros_like(buffer.rewards).to(device)
            last_gae_lam = 0

            for t in reversed(range(buffer.buffer_size)):
                if t == buffer.buffer_size - 1:
                    next_non_terminal = 1.0 - current_dones
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - buffer.dones[t]
                    next_values = buffer.values[t + 1]

                # the GAE math
                delta = buffer.rewards[t] + GAMMA * next_values * next_non_terminal - buffer.values[t]
                advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam

            returns = advantages + buffer.values

            # flattening the data for PPO
            # ex. 2048 steps * 10 agents into 2048 flat rows

            b_states = buffer.states.view(-1, STATE_SPACE).to(device)
            b_actions = buffer.actions.view(-1, ACTION_SPACE).to(device)
            b_logprobs = buffer.logprobs.view(-1).to(device)
            b_advantages = advantages.view(-1).to(device)
            b_returns = returns.view(-1).to(device)

            # PPO requires tracking the total batch size to create mini-batches 
            b_size = buffer.buffer_size * nr_of_agents
            b_inds = torch.arange(b_size)

            for epoch in range(PPO_EPOCHS):
                b_inds = b_inds[torch.randperm(b_size)]
                for start in range(0, b_size, MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]

                    # Get the current predictions for the OLD states and OLD actions
                    _, new_logprob, entropy, new_value = model.get_action_and_value(b_states[mb_inds], b_actions[mb_inds])
                    new_value = new_value.squeeze()

                    # advantage normalization (crucial for stability)
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # ratio calculation
                    # Because we use Log Probabilities, subtracting them is mathematically 
                    # identical to dividing normal probabilities, then we take exp() to undo the log.
                    logratio = new_logprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # actor loss (clipped surrogate objective)
                    pg_loss1 = mb_advantages * ratio
                    pg_loss2 = mb_advantages * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                    pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                    # critic loss (MSE)
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                    # entropy bonus
                    entropy_loss = entropy.mean()

                    # total loss
                    loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                    optimizer.zero_grad()
                    loss.backward()

                    # clip gradients to prevent the network weights from exploding
                    clip_grad_norm_(model.parameters(), 0.5)

                    optimizer.step()


            buffer.reset()

except KeyboardInterrupt:
    print('stopped by user')
except Exception as e:
    print(f'an error occured: {e}')

finally:
    env.close()  # Unfreezes Unity gracefully
    
    # 1. Prepare for export
    model.eval()
    dummy_input = torch.randn(1, STATE_SPACE, device=device)
    onnx_filename = "DriverNet.onnx"

    # 2. Export natively via PyTorch
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["obs_0"],
        # Map exactly to the 4 outputs returned by our new forward() method
        output_names=[
            "continuous_actions",
            "version_number",
            "memory_size",
            "continuous_action_output_shape"
        ],
        dynamic_axes={
            "obs_0": {0: "batch_size"}, 
            "continuous_actions": {0: "batch_size"}
        },
    )

    print(f"Model successfully saved to {onnx_filename}!")
    print("Disconnected safely.")