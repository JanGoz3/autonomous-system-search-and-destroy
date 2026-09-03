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
import time

# use non-interactive backend for headless plotting.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

class TrainingTracker:
    def __init__(self, window_size = 100):
        self.episode_rewards = deque(maxlen=window_size)
        self.history_steps = []
        self.history_mean_rewards = []
        self.history_std_rewards = []

    def add_reward(self, reward):
        self.episode_rewards.append(reward)

    def get_stats(self):
        if len(self.episode_rewards) == 0:
            return 0.0, 0.0
        return np.mean(self.episode_rewards), np.std(self.episode_rewards)

    def log_progress(self, total_steps):
        mean_rew, std_rew = self.get_stats()
        self.history_steps.append(total_steps)
        self.history_mean_rewards.append(mean_rew)
        self.history_std_rewards.append(std_rew)
        self.save_plot()

    def save_plot(self, filename="training_progress.png"):
        if len(self.history_steps) == 0:
            return

        plt.figure(figsize=(10, 5))
        steps = np.array(self.history_steps)
        means = np.array(self.history_mean_rewards)
        stds = np.array(self.history_std_rewards)

        # Plot mean reward curve
        plt.plot(steps, means, label='Mean Reward (Last 100)', color='b', linewidth=2)
        # Plot standard deviation shading band
        plt.fill_between(steps, means - stds, means + stds, color='b', alpha=0.15, label='Std Deviation')

        plt.title('Autonomous Car RL Training Progress')
        plt.xlabel('Total Agent Steps')
        plt.ylabel('Reward')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left')
        plt.tight_layout()

        # Save overwrite file
        plt.savefig(filename)
        plt.close()


print('running... waiting for connection with Unity simulation')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
GAE_LAMBDA = 0.95
STACKED_VECTORS = 3
STATE_SPACE = 40
ACTION_SPACE = 4
# LEARNING_RATE = 3e-4
LEARNING_RATE = 5e-5
PPO_EPOCHS = 4
MINIBATCH_SIZE = 1024
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
CHECKPOINT_FILE = "driver_checkpoint.pth"
BEST_CHECKPOINT_FILE = "best_driver_checkpoint.pth"

engine_channel = EngineConfigurationChannel()
engine_channel.set_configuration_parameters(time_scale=5.0)
env = UnityEnvironment(file_name=None, side_channels=[engine_channel])
model = DriverNet(in_features=STATE_SPACE * STACKED_VECTORS, out_features=ACTION_SPACE).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
tracker = TrainingTracker(window_size=100)

# ==========================================
# LOAD CHECKPOINT (IF IT EXISTS)
# ==========================================
start_steps = 0
if os.path.exists(CHECKPOINT_FILE):
    print(f"Loading checkpoint from {CHECKPOINT_FILE}...")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE
    
    if 'tracker_history' in checkpoint:
        tracker.history_steps = checkpoint['tracker_history']['steps']
        tracker.history_mean_rewards = checkpoint['tracker_history']['means']
        tracker.history_std_rewards = checkpoint['tracker_history']['stds']
        if len(tracker.history_steps) > 0:
            start_steps = tracker.history_steps[-1]
            
    print(f"Resuming training from step {start_steps}!")
else:
    print("No checkpoint found. Starting training from scratch.")

best_mean_reward = max(tracker.history_mean_rewards) if len(tracker.history_mean_rewards) > 0 else -float('inf')
# ==========================================

try:
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]   

    print("Connected! Running... Press Ctrl+C to stop.")
    start_time = time.time()
    # ask unity for the current state of the agents
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    nr_of_agents = len(decision_steps)
    current_episode_rewards = np.zeros(nr_of_agents)
    total_env_steps = start_steps
    
    # This assigns each Unity agent a permanent "row index" (0 to 9) in our PyTorch arrays. 
    id_to_idx = {agent_id: i for i, agent_id in enumerate(decision_steps.agent_id)}

    buffer = RolloutBuffer(
        nr_of_agents=nr_of_agents, 
        device=device, 
        action_space= ACTION_SPACE, 
        state_space = STATE_SPACE * STACKED_VECTORS,
        buffer_size=10240
    )

    # TODO: optionally add retrieving from the environment action space and state space

    state_tensor = torch.zeros((nr_of_agents, STATE_SPACE * STACKED_VECTORS), dtype=torch.float32).to(device)
    for i, agent_id in enumerate(decision_steps.agent_id):
        if agent_id in id_to_idx:
            state_tensor[id_to_idx[agent_id]] = torch.tensor(decision_steps.obs[0][i], dtype=torch.float32).to(device)
    for i, agent_id in enumerate(terminal_steps.agent_id):
        if agent_id in id_to_idx:
            state_tensor[id_to_idx[agent_id]] = torch.tensor(terminal_steps.obs[0][i], dtype=torch.float32).to(device)

    while True:
        # here the agent is just playing, not learning yet
        with torch.no_grad():
            action_tensor, log_prob, entropy, value = model.get_action_and_value(state_tensor)
            value = value.flatten()
            log_prob = log_prob.flatten()

        # unity will throw error if we send action to a dead agent so we filter those out
        active_actions = torch.zeros((len(decision_steps), ACTION_SPACE), dtype=torch.float32).to(device)
        for i, agent_id in enumerate(decision_steps.agent_id):
            if agent_id in id_to_idx:
                active_actions[i] = action_tensor[id_to_idx[agent_id]]

        action_numpy = active_actions.cpu().numpy()
        action_tuple = ActionTuple(continuous=action_numpy)
        env.set_actions(behavior_name, action_tuple)
        
        env.step()  # Ticks the physics loop forward 1 frame

        # get new outcomes from the step we just took
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        next_state_tensor = torch.zeros((nr_of_agents, STATE_SPACE * STACKED_VECTORS), dtype=torch.float32).to(device)

        current_rewards = torch.zeros(nr_of_agents).to(device)
        current_dones = torch.zeros(nr_of_agents).to(device)

        # match rewards to agents that are still playing
        for i, agent_id in enumerate(decision_steps.agent_id):
            if agent_id in id_to_idx:
                idx = id_to_idx[agent_id]
                reward = float(decision_steps.reward[i])
                current_rewards[idx] = reward
                current_dones[idx] = 0.0
                next_state_tensor[idx] = torch.tensor(decision_steps.obs[0][i], dtype=torch.float32).to(device)
                current_episode_rewards[idx] += reward

        # match rewards to agents that crashed/finished
        for i, agent_id in enumerate(terminal_steps.agent_id):
            if agent_id in id_to_idx:
                idx = id_to_idx[agent_id]
                reward = float(terminal_steps.reward[i])
                current_rewards[idx] = reward
                current_dones[idx] = 1.0
                next_state_tensor[idx] = torch.tensor(terminal_steps.obs[0][i], dtype=torch.float32).to(device)
                current_episode_rewards[idx] += reward
                tracker.add_reward(current_episode_rewards[idx])
                current_episode_rewards[idx] = 0.0

        buffer.insert(state_tensor, action_tensor, log_prob, value, current_rewards, current_dones)
        state_tensor = next_state_tensor

        total_env_steps += nr_of_agents

        if buffer.step_counter == buffer.buffer_size:
            tracker.log_progress(total_env_steps)
            mean_rew, std_rew = tracker.get_stats()
            current_time = time.time()
            passed_time = current_time - start_time
            print(f"==================================================")
            print(f"Total Agent Steps: {total_env_steps}")
            print(f"Mean Reward (Last 100 episodes): {mean_rew:.2f}")
            print(f"Std Reward  (Last 100 episodes): {std_rew:.2f}")
            print(f"Time Elapsed: {passed_time:.2f} seconds")
            print(f"==================================================")
            print("buffer full. Training.")

            with torch.no_grad():
                next_value = model.get_value(state_tensor).flatten()

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

            b_states = buffer.states.view(-1, STATE_SPACE * STACKED_VECTORS)
            b_actions = buffer.actions.view(-1, ACTION_SPACE)
            b_logprobs = buffer.logprobs.view(-1)
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

            # ==========================================
            # SAVE CHECKPOINT
            # ==========================================
            save_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tracker_history': {
                    'steps': tracker.history_steps,
                    'means': tracker.history_mean_rewards,
                    'stds': tracker.history_std_rewards
                }
            }
            torch.save(save_data, CHECKPOINT_FILE)
            print(f"Checkpoint successfully saved to {CHECKPOINT_FILE}")

            # Save a dedicated copy if we beat our personal best mean reward
            if mean_rew > best_mean_reward:
                best_mean_reward = mean_rew
                torch.save(save_data, BEST_CHECKPOINT_FILE)
                print(f"*** NEW ALL-TIME BEST MODEL. Saved to {BEST_CHECKPOINT_FILE} (Reward: {mean_rew:.2f}) ***")

            # ==========================================


except KeyboardInterrupt:
    print('stopped by user')
except Exception as e:
    print(f'an error occured: {e}')

finally:
    env.close()  # Unfreezes Unity gracefully
    
    # 1. Prepare for export
    model.eval()
    dummy_input = torch.randn(1, STATE_SPACE * STACKED_VECTORS, device=device)
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