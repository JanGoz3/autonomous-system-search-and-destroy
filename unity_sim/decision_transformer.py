import numpy as np
import torch
import torch.nn as nn


def discount_cumsum(x, gamma=1.0):

    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=128, n_layer=3, n_head=1,
                 max_ep_len=1000, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim), nn.Tanh()
        )

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask):

        B, T, _ = states.shape
        device = states.device

        time_emb = self.embed_timestep(timesteps)
        state_emb = self.embed_state(states) + time_emb
        action_emb = self.embed_action(actions) + time_emb
        return_emb = self.embed_return(returns_to_go) + time_emb

        stacked = torch.stack((return_emb, state_emb, action_emb), dim=2)  # (B, T, 3, H)
        stacked = stacked.reshape(B, 3 * T, self.hidden_size)
        stacked = self.embed_ln(stacked)

        causal_mask = torch.triu(
            torch.ones(3 * T, 3 * T, device=device, dtype=torch.bool), diagonal=1
        )

        key_padding_mask = (attention_mask == 0).repeat_interleave(3, dim=1)  # (B, 3T), True = ignoruj

        x = self.transformer(
            stacked, mask=causal_mask, src_key_padding_mask=key_padding_mask
        )

        x = x.reshape(B, T, 3, self.hidden_size)

        action_preds = self.predict_action(x[:, :, 1])  # (B, T, act_dim)

        return action_preds

    @torch.no_grad()
    def get_action(self, states, actions, returns_to_go, timesteps, max_length, device):

        state_dim = states.shape[-1]
        act_dim = actions.shape[-1]

        states = states.reshape(1, -1, state_dim)
        actions = actions.reshape(1, -1, act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -max_length:]
        actions = actions[:, -max_length:]
        returns_to_go = returns_to_go[:, -max_length:]
        timesteps = timesteps[:, -max_length:]

        tlen = states.shape[1]
        pad = max_length - tlen

        attention_mask = torch.cat(
            [torch.zeros(pad, device=device), torch.ones(tlen, device=device)]
        ).reshape(1, -1)

        states = torch.cat(
            [torch.zeros((1, pad, state_dim), device=device), states], dim=1
        )
        actions = torch.cat(
            [torch.zeros((1, pad, act_dim), device=device), actions], dim=1
        )
        returns_to_go = torch.cat(
            [torch.zeros((1, pad, 1), device=device), returns_to_go], dim=1
        )
        timesteps = torch.cat(
            [torch.zeros((1, pad), dtype=torch.long, device=device), timesteps], dim=1
        )

        action_preds = self.forward(states, actions, returns_to_go, timesteps, attention_mask)
        return action_preds[0, -1]