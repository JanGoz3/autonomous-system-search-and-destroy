import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mha.set_fastpath_enabled(False)

def discount_cumsum(x, gamma=1.0):
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=128, n_layer=3, n_head=1,
                 max_ep_len=1000, dropout=0.1,
                 action_head="continuous", n_dir_bins=36, mag_weight=1.0,
                 label_smooth=0.15):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.action_head = action_head
        self.n_dir_bins = n_dir_bins
        self.mag_weight = mag_weight
        self.label_smooth = label_smooth

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_head, dim_feedforward=4 * hidden_size,
            dropout=dropout, activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layer)

        if action_head == "continuous":
            self.predict_action = nn.Sequential(nn.Linear(hidden_size, act_dim), nn.Tanh())
        elif action_head == "discrete":
            self.predict_dir = nn.Linear(hidden_size, n_dir_bins)
            self.predict_mag = nn.Linear(hidden_size, 1)
            centers = (torch.arange(n_dir_bins, dtype=torch.float32) + 0.5) \
                      * (2 * np.pi / n_dir_bins) - np.pi
            self.register_buffer("bin_centers", centers)
        else:
            raise ValueError(action_head)

    def _hidden(self, states, actions, returns_to_go, timesteps, attention_mask):
        B, T, _ = states.shape
        time_emb = self.embed_timestep(timesteps)
        stacked = torch.stack((self.embed_return(returns_to_go) + time_emb,
                               self.embed_state(states) + time_emb,
                               self.embed_action(actions) + time_emb), dim=2)
        stacked = self.embed_ln(stacked.reshape(B, 3 * T, self.hidden_size))

        causal = torch.triu(torch.ones(3 * T, 3 * T, device=states.device,
                                       dtype=torch.bool), diagonal=1)
        kpm = (attention_mask == 0).repeat_interleave(3, dim=1)
        x = self.transformer(stacked, mask=causal, src_key_padding_mask=kpm)
        return x.reshape(B, T, 3, self.hidden_size)[:, :, 1]      # token stanu

    def _vector_from_heads(self, h):
        logits = self.predict_dir(h)
        mag = F.softplus(self.predict_mag(h))                      # (B,T,1)
        ang = self.bin_centers[logits.argmax(dim=-1)]              # (B,T)
        return torch.cat([mag * torch.sin(ang).unsqueeze(-1),
                          mag * torch.cos(ang).unsqueeze(-1)], dim=-1)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask):
        """Zawsze zwraca WEKTORY (B, T, act_dim), zeby cala reszta pipeline'u
        (ewaluacja, autoregresja, eksport) dzialala bez zmian."""
        h = self._hidden(states, actions, returns_to_go, timesteps, attention_mask)
        if self.action_head == "continuous":
            return self.predict_action(h)
        return self._vector_from_heads(h)

    def compute_loss(self, states, actions_in, returns_to_go, timesteps,
                     attention_mask, target_actions, loss_mask):
        h = self._hidden(states, actions_in, returns_to_go, timesteps, attention_mask)
        m = loss_mask
        denom = m.sum().clamp(min=1.0)

        if self.action_head == "continuous":
            err = ((self.predict_action(h) - target_actions) ** 2).sum(-1)
            return (err * m).sum() / denom

        B, T, n = h.shape[0], h.shape[1], self.n_dir_bins
        logits = self.predict_dir(h)
        mag = F.softplus(self.predict_mag(h)).squeeze(-1)

        tgt_ang = torch.atan2(target_actions[..., 0], target_actions[..., 1])
        tgt_bin = torch.floor((tgt_ang + np.pi) / (2 * np.pi / n)).long().clamp(0, n - 1)

        s = self.label_smooth
        soft = torch.zeros(B, T, n, device=h.device)
        soft.scatter_(2, tgt_bin.unsqueeze(-1), 1.0 - 2 * s)
        soft.scatter_add_(2, ((tgt_bin - 1) % n).unsqueeze(-1),
                          torch.full_like(soft[..., :1], s))
        soft.scatter_add_(2, ((tgt_bin + 1) % n).unsqueeze(-1),
                          torch.full_like(soft[..., :1], s))

        ce = -(soft * F.log_softmax(logits, dim=-1)).sum(-1)
        mag_err = (mag - target_actions.norm(dim=-1)) ** 2
        return ((ce + self.mag_weight * mag_err) * m).sum() / denom

    @torch.no_grad()
    def get_action(self, states, actions, returns_to_go, timesteps, max_length, device):
        sd, ad = states.shape[-1], actions.shape[-1]
        states = states.reshape(1, -1, sd)[:, -max_length:]
        actions = actions.reshape(1, -1, ad)[:, -max_length:]
        returns_to_go = returns_to_go.reshape(1, -1, 1)[:, -max_length:]
        timesteps = timesteps.reshape(1, -1)[:, -max_length:]

        tlen = states.shape[1]
        pad = max_length - tlen
        z = lambda *sh: torch.zeros(*sh, device=device)

        attention_mask = torch.cat([z(pad), torch.ones(tlen, device=device)]).reshape(1, -1)
        states = torch.cat([z(1, pad, sd), states], dim=1)
        actions = torch.cat([z(1, pad, ad), actions], dim=1)
        returns_to_go = torch.cat([z(1, pad, 1), returns_to_go], dim=1)
        timesteps = torch.cat([torch.zeros((1, pad), dtype=torch.long, device=device),
                               timesteps], dim=1)

        return self.forward(states, actions, returns_to_go, timesteps, attention_mask)[0, -1]