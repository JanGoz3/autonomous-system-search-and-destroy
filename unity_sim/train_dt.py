import pickle

import numpy as np
import torch
import torch.nn as nn

from decision_transformer import DecisionTransformer

torch.backends.mha.set_fastpath_enabled(False)

DATASET_FILE = "dt_dataset.pkl"
CHECKPOINT_FILE = "dt_checkpoint.pt"

CONTEXT_LENGTH = 20
HIDDEN_SIZE = 128
N_LAYER = 3
N_HEAD = 4
DROPOUT = 0.1

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_NORM_CLIP = 0.25
WARMUP_ITERS = 200

NUM_TRAIN_ITERS = 3000
LOG_EVERY = 100
VAL_EVERY = 100
VAL_BATCHES = 8

NUM_HELDOUT_EPISODES = 4
SPLIT_SEED = 42

USE_YAW_SINCOS = False


ZERO_ACTIONS_IN_CONTEXT = True

ACTION_HEAD = "discrete"
N_DIR_BINS = 36          # 36 binow = 10 stopni na bin
MAG_WEIGHT = 1.0         # waga regresji dlugosci wzgledem cross-entropy
LABEL_SMOOTH = 0.15      # ile masy oddac na kazdy sasiedni bin

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data["trajectories"], data.get("action_scale", 1.0)
    print("UWAGA: stary format datasetu (lista). Brak maski 'valid' i action_scale.")
    return data, 1.0


def apply_yaw_sincos(trajectories):
    for t in trajectories:
        s = t["states"]
        yaw = np.radians(s[:, 2].astype(np.float64))
        t["states"] = np.column_stack([
            s[:, 0:2],
            np.sin(yaw).astype(np.float32),
            np.cos(yaw).astype(np.float32),
            s[:, 3:],
        ]).astype(np.float32)


def get_valid(traj):
    if "valid" in traj:
        return traj["valid"].astype(np.float32)
    return np.ones(traj["states"].shape[0], dtype=np.float32)


def compute_state_normalization(trajectories):
    all_states = np.concatenate([t["states"] for t in trajectories], axis=0)
    return (all_states.mean(axis=0).astype(np.float32),
            (all_states.std(axis=0) + 1e-6).astype(np.float32))


def compute_return_scale(trajectories):
    rtg = np.concatenate([t["returns_to_go"] for t in trajectories])
    scale = float(np.abs(rtg).std())
    return max(scale, 1e-3)


def get_batch(trajectories, batch_size, K, state_dim, act_dim,
              state_mean, state_std, return_scale, max_ep_len, device, rng):
    lengths = np.array([t["states"].shape[0] for t in trajectories], dtype=np.float64)
    p_sample = lengths / lengths.sum()
    batch_inds = rng.choice(len(trajectories), size=batch_size, p=p_sample)

    s_l, a_l, ain_l, r_l, t_l, m_l, v_l = [], [], [], [], [], [], []

    for idx in batch_inds:
        traj = trajectories[idx]
        traj_len = traj["states"].shape[0]
        si = rng.randint(0, traj_len - 1)
        end = min(si + K, traj_len)

        s = (traj["states"][si:end] - state_mean) / state_std
        a = traj["actions"][si:end]
        r = traj["returns_to_go"][si:end].reshape(-1, 1) / return_scale
        v = get_valid(traj)[si:end]
        ts = np.clip(np.arange(si, end), 0, max_ep_len - 1)

        tlen = s.shape[0]
        pad = K - tlen
        z = lambda shape, dt=np.float32: np.zeros(shape, dtype=dt)

        a_pad = np.concatenate([z((pad, act_dim)), a])
        s_l.append(np.concatenate([z((pad, state_dim)), s]))
        a_l.append(a_pad)
        ain_l.append(np.zeros_like(a_pad) if ZERO_ACTIONS_IN_CONTEXT else a_pad)
        r_l.append(np.concatenate([z((pad, 1)), r]))
        t_l.append(np.concatenate([z((pad,), np.int64), ts]))
        m_l.append(np.concatenate([z((pad,)), np.ones(tlen, dtype=np.float32)]))
        v_l.append(np.concatenate([z((pad,)), v]))

    to = lambda arr, dt: torch.tensor(np.stack(arr), dtype=dt, device=device)
    return (to(s_l, torch.float32), to(a_l, torch.float32), to(ain_l, torch.float32),
            to(r_l, torch.float32), to(t_l, torch.long), to(m_l, torch.float32),
            to(v_l, torch.float32))


def masked_mse(preds, actions, attn_mask, valid_mask):
    m = (attn_mask * valid_mask).unsqueeze(-1)
    denom = m.sum().clamp(min=1.0)
    return (((preds - actions) * m) ** 2).sum() / denom


def train_once(split_seed=SPLIT_SEED, num_iters=NUM_TRAIN_ITERS, verbose=True,
               save_path=CHECKPOINT_FILE):
    trajectories, action_scale = load_dataset(DATASET_FILE)
    if USE_YAW_SINCOS:
        apply_yaw_sincos(trajectories)

    groups = sorted({t.get("group", t["source_file"]) for t in trajectories})
    rng_split = np.random.RandomState(split_seed)
    gidx = rng_split.permutation(len(groups))
    n_val = min(NUM_HELDOUT_EPISODES, max(1, len(groups) // 8))
    held_out = sorted(groups[i] for i in gidx[:n_val])
    hset = set(held_out)
    val_traj = [t for t in trajectories if t.get("group", t["source_file"]) in hset]
    train_traj = [t for t in trajectories if t.get("group", t["source_file"]) not in hset]

    state_dim = trajectories[0]["states"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    max_ep_len = max(t["states"].shape[0] for t in trajectories) + 10

    state_mean, state_std = compute_state_normalization(train_traj)
    return_scale = compute_return_scale(train_traj)

    if verbose:
        print(f"Urzadzenie: {DEVICE}")
        print(f"Trening: {len(train_traj)} sekwencji, walidacja: {len(val_traj)} "
              f"({len(groups)} grup, {n_val} odlozonych)")
        print(f"Odlozone: {held_out}")
        print(f"state_dim={state_dim}, act_dim={act_dim}, "
              f"action_scale={action_scale:.4f} m, return_scale={return_scale:.2f}")

    model = DecisionTransformer(
        state_dim=state_dim, act_dim=act_dim, hidden_size=HIDDEN_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, max_ep_len=max_ep_len, dropout=DROPOUT,
        action_head=ACTION_HEAD, n_dir_bins=N_DIR_BINS,
        mag_weight=MAG_WEIGHT, label_smooth=LABEL_SMOOTH,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min((it + 1) / WARMUP_ITERS, 1.0))

    rng = np.random.RandomState(split_seed + 1000)
    batch_args = (BATCH_SIZE, CONTEXT_LENGTH, state_dim, act_dim,
                  state_mean, state_std, return_scale, max_ep_len, DEVICE)

    best_val, best_state = float("inf"), None

    for it in range(1, num_iters + 1):
        model.train()
        s, a, a_in, r, ts, m, v = get_batch(train_traj, *batch_args, rng)
        loss = model.compute_loss(s, a_in, r, ts, m, a, m * v)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
        opt.step()
        sched.step()

        if it % VAL_EVERY == 0 or it == 1:
            model.eval()
            vs = []
            with torch.no_grad():
                for _ in range(VAL_BATCHES):
                    s, a, a_in, r, ts, m, v = get_batch(val_traj, *batch_args, rng)
                    vs.append(model.compute_loss(s, a_in, r, ts, m, a, m * v).item())
            vloss = float(np.mean(vs))
            if vloss < best_val:
                best_val = vloss
                best_state = {k: t.detach().cpu().clone()
                              for k, t in model.state_dict().items()}
            if verbose:
                print(f"[{it:5d}/{num_iters}] train={loss.item():.5f}  "
                      f"val={vloss:.5f}  (best={best_val:.5f})")
        elif verbose and it % LOG_EVERY == 0:
            print(f"[{it:5d}/{num_iters}] train={loss.item():.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "held_out_files": held_out,
        "best_val_loss": best_val,
        "config": {
            "state_dim": state_dim, "act_dim": act_dim,
            "hidden_size": HIDDEN_SIZE, "n_layer": N_LAYER, "n_head": N_HEAD,
            "max_ep_len": max_ep_len, "context_length": CONTEXT_LENGTH,
            "return_scale": return_scale,
            "action_scale": action_scale,
            "use_yaw_sincos": USE_YAW_SINCOS,
            "zero_actions": ZERO_ACTIONS_IN_CONTEXT,
            "action_head": ACTION_HEAD,
            "n_dir_bins": N_DIR_BINS,
            "mag_weight": MAG_WEIGHT,
            "label_smooth": LABEL_SMOOTH,
        },
    }
    if save_path:
        torch.save(ckpt, save_path)
        if verbose:
            print(f"\nZapisano checkpoint (najlepszy val={best_val:.5f}) do {save_path}")
    return ckpt, best_val


if __name__ == "__main__":
    train_once()