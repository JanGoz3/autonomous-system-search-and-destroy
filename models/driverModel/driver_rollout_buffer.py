import torch

class RolloutBuffer:
    def __init__(self, buffer_size = 2048, nr_of_agents = 1, action_space = 2, state_space = 8, device = "cpu"):
        self.states = torch.zeros((buffer_size, nr_of_agents, state_space)).to(device)
        self.actions = torch.zeros((buffer_size, nr_of_agents, action_space)).to(device)

        self.logprobs = torch.zeros((buffer_size, nr_of_agents)).to(device)
        self.values = torch.zeros((buffer_size, nr_of_agents)).to(device)
        self.rewards = torch.zeros((buffer_size, nr_of_agents)).to(device)
        self.dones = torch.zeros((buffer_size, nr_of_agents)).to(device)

        self.masks = torch.zeros((buffer_size, nr_of_agents)).to(device)

        self.step_counter = 0
        self.buffer_size = buffer_size


    def insert(self, state, action, log_prob, value, reward, done, mask):

        self.states[self.step_counter] = state
        self.actions[self.step_counter] = action
        self.logprobs[self.step_counter] = log_prob
        self.values[self.step_counter] = value
        self.rewards[self.step_counter] = reward
        self.dones[self.step_counter] = done
        self.masks[self.step_counter] = mask

        self.step_counter += 1

    def reset(self):
        self.step_counter = 0

        