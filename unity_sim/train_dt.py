import pickle
import numpy as np
import torch
import torch.nn as nn
from decision_transformer import DecisionTransformer

torch.backends.mha.set_fastpath_enabled(False)


DATASET_FILE = "dt_dataset.pkl"
CHECKPOINT_FILE = "dt_checkpoint.pt"

CONTEXT_LENGTH = 20        # K - dlugosc kontekstu (ile krokow historii widzi model)
HIDDEN_SIZE = 128
N_LAYER = 3
N_HEAD = 4
DROPOUT = 0.1

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_NORM_CLIP = 0.25

NUM_TRAIN_ITERS = 3000      # liczba krokow gradientowych (nie epok - kazdy krok to nowy losowy batch)
LOG_EVERY = 100
VAL_EVERY = 100

NUM_HELDOUT_EPISODES = 2    # ile epizodow odkladamy jako zbior walidacyjny
SPLIT_SEED = 42              # ziarno losowosci dla podzialu na train/val

RETURN_SCALE = 50.0         # dzielimy return-to-go przez ta wartosc, zeby trzymac wejscie w rozsadnej skali

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(path):
    with open(path, "rb") as f:
        trajectories = pickle.load(f)
    return trajectories


def compute_state_normalization(trajectories):
    all_states = np.concatenate([t["states"] for t in trajectories], axis=0)
    mean = all_states.mean(axis=0)
    std = all_states.std(axis=0) + 1e-6  # unikamy dzielenia przez 0 dla stalych kolumn
    return mean.astype(np.float32), std.astype(np.float32)


def get_batch(trajectories, batch_size, K, state_dim, act_dim,
              state_mean, state_std, max_ep_len, device):
    lengths = np.array([t["states"].shape[0] for t in trajectories], dtype=np.float32)
    p_sample = lengths / lengths.sum()  # dluzsze trajektorie losowane czesciej

    batch_inds = np.random.choice(len(trajectories), size=batch_size, p=p_sample)

    s_list, a_list, rtg_list, t_list, mask_list = [], [], [], [], []

    for idx in batch_inds:
        traj = trajectories[idx]
        traj_len = traj["states"].shape[0]

        si = np.random.randint(0, traj_len - 1)
        end = min(si + K, traj_len)

        s = traj["states"][si:end]
        a = traj["actions"][si:end]
        rtg = traj["returns_to_go"][si:end].reshape(-1, 1) / RETURN_SCALE
        ts = np.arange(si, end)
        ts = np.clip(ts, 0, max_ep_len - 1)

        tlen = s.shape[0]
        pad = K - tlen

        s = (s - state_mean) / state_std
        s = np.concatenate([np.zeros((pad, state_dim), dtype=np.float32), s], axis=0)
        a = np.concatenate([np.zeros((pad, act_dim), dtype=np.float32), a], axis=0)
        rtg = np.concatenate([np.zeros((pad, 1), dtype=np.float32), rtg], axis=0)
        ts = np.concatenate([np.zeros((pad,), dtype=np.int64), ts], axis=0)
        mask = np.concatenate([np.zeros((pad,), dtype=np.float32), np.ones((tlen,), dtype=np.float32)], axis=0)

        s_list.append(s)
        a_list.append(a)
        rtg_list.append(rtg)
        t_list.append(ts)
        mask_list.append(mask)

    s_batch = torch.tensor(np.stack(s_list), dtype=torch.float32, device=device)
    a_batch = torch.tensor(np.stack(a_list), dtype=torch.float32, device=device)
    rtg_batch = torch.tensor(np.stack(rtg_list), dtype=torch.float32, device=device)
    t_batch = torch.tensor(np.stack(t_list), dtype=torch.long, device=device)
    mask_batch = torch.tensor(np.stack(mask_list), dtype=torch.float32, device=device)

    return s_batch, a_batch, rtg_batch, t_batch, mask_batch

def main():
    print(f"Urzadzenie: {DEVICE}")

    trajectories = load_dataset(DATASET_FILE)
    print(f"Wczytano {len(trajectories)} trajektorii z {DATASET_FILE}")

    rng = np.random.RandomState(SPLIT_SEED)
    shuffled_idx = rng.permutation(len(trajectories))

    n_val = min(NUM_HELDOUT_EPISODES, max(0, len(trajectories) - 1))
    val_idx = shuffled_idx[:n_val]
    train_idx = shuffled_idx[n_val:]

    train_trajectories = [trajectories[i] for i in train_idx]
    val_trajectories = [trajectories[i] for i in val_idx]
    held_out_files = [t["source_file"] for t in val_trajectories]

    print(f"Trening: {len(train_trajectories)} trajektorii, Walidacja: {len(val_trajectories)} trajektorii")
    print(f"Odlozone do walidacji pliki: {held_out_files}")

    state_dim = trajectories[0]["states"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    max_ep_len = max(t["states"].shape[0] for t in trajectories) + 10

    print(f"state_dim={state_dim}, act_dim={act_dim}, max_ep_len={max_ep_len}")

    state_mean, state_std = compute_state_normalization(train_trajectories)

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=HIDDEN_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        max_ep_len=max_ep_len,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for iteration in range(1, NUM_TRAIN_ITERS + 1):
        model.train()
        states, actions, rtg, timesteps, mask = get_batch(
            train_trajectories, BATCH_SIZE, CONTEXT_LENGTH, state_dim, act_dim,
            state_mean, state_std, max_ep_len, DEVICE,
        )

        action_preds = model(states, actions, rtg, timesteps, mask)

        loss_mask = mask.unsqueeze(-1)  # (B, K, 1)
        diff = (action_preds - actions) * loss_mask
        loss = (diff ** 2).sum() / loss_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
        optimizer.step()

        if iteration % LOG_EVERY == 0 or iteration == 1:
            print(f"[{iteration:5d}/{NUM_TRAIN_ITERS}] train_loss={loss.item():.6f}")

        if len(val_trajectories) > 0 and (iteration % VAL_EVERY == 0 or iteration == 1):
            model.eval()
            with torch.no_grad():
                v_states, v_actions, v_rtg, v_timesteps, v_mask = get_batch(
                    val_trajectories, min(BATCH_SIZE, len(val_trajectories) * 4), CONTEXT_LENGTH,
                    state_dim, act_dim, state_mean, state_std, max_ep_len, DEVICE,
                )
                v_preds = model(v_states, v_actions, v_rtg, v_timesteps, v_mask)
                v_loss_mask = v_mask.unsqueeze(-1)
                v_diff = (v_preds - v_actions) * v_loss_mask
                v_loss = (v_diff ** 2).sum() / v_loss_mask.sum()
            print(f"           [{iteration:5d}/{NUM_TRAIN_ITERS}] val_loss={v_loss.item():.6f}")

    print("\nTrening zakonczony.")

    torch.save({
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "held_out_files": held_out_files,
        "config": {
            "state_dim": state_dim,
            "act_dim": act_dim,
            "hidden_size": HIDDEN_SIZE,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "max_ep_len": max_ep_len,
            "context_length": CONTEXT_LENGTH,
            "return_scale": RETURN_SCALE,
        },
    }, CHECKPOINT_FILE)

    print(f"Zapisano checkpoint do {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()