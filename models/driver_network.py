import torch.nn as nn
import torch
from torch.distributions.normal import Normal

class DriverNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.log_std = nn.Parameter(torch.zeros(1, out_features))

        self.actor_block = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, out_features),
            nn.Tanh()
        )

        self.critic_block = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_value(self, state):
        return self.critic_block(state)

    def get_action_and_value(self, state, action=None):
        action_mean = self.actor_block(state)

        # converting log standard deviation to actual standard deviation (must be positive)
        action_std = torch.exp(self.log_std)

        probs = Normal(action_mean, action_std)

        # if we are playing (no action provided), sample a new one
        if action is None:
            action = probs.sample()

        # PPO needs joint probability of all actions so we sum them together along the action dimmension
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        # value prediction from the critic
        value = self.critic_block(state)

        return action, log_prob, entropy, value