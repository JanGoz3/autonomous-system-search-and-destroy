import pickle

import numpy as np
import torch
import torch.nn as nn

torch.backends.mha.set_fastpath_enabled(False)

from decision_transformer import DecisionTransformer


DATASET_FILE = "dt_dataset.pkl"
SEEDS_TO_TEST = [0, 1, 2, 3, 4]   # kilka roznych podzialow train/val
CONTEXT_LENGTH = 20
HIDDEN_SIZE = 128
N_LAYER = 3
N_HEAD = 4
DROPOUT = 0.1

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_NORM_CLIP = 0.25

NUM_TRAIN_ITERS = 1500
NUM_HELDOUT_EPISODES = 2

RETURN_SCALE = 50.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_state_normalization(trajectories):
    all_states = np.concatenate([t["states"] for t in trajectories], axis=0)
    mean = all_states.mean(axis=0)
    std = all_states.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def get_batch(trajectories, batch_size, K, state_dim, act_dim, state_mean, state_std, max_ep_len, device):
    lengths = np.array([t["states"].shape[0] for t in trajectories], dtype=np.float32)
    p_sample = lengths / lengths.sum()
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
        ts = np.clip(np.arange(si, end), 0, max_ep_len - 1)

        tlen = s.shape[0]
        pad = K - tlen

        s = (s - state_mean) / state_std
        s = np.concatenate([np.zeros((pad, state_dim), dtype=np.float32), s], axis=0)
        a = np.concatenate([np.zeros((pad, act_dim), dtype=np.float32), a], axis=0)
        rtg = np.concatenate([np.zeros((pad, 1), dtype=np.float32), rtg], axis=0)
        ts = np.concatenate([np.zeros((pad,), dtype=np.int64), ts], axis=0)
        mask = np.concatenate([np.zeros((pad,), dtype=np.float32), np.ones((tlen,), dtype=np.float32)], axis=0)

        s_list.append(s); a_list.append(a); rtg_list.append(rtg); t_list.append(ts); mask_list.append(mask)

    return (
        torch.tensor(np.stack(s_list), dtype=torch.float32, device=device),
        torch.tensor(np.stack(a_list), dtype=torch.float32, device=device),
        torch.tensor(np.stack(rtg_list), dtype=torch.float32, device=device),
        torch.tensor(np.stack(t_list), dtype=torch.long, device=device),
        torch.tensor(np.stack(mask_list), dtype=torch.float32, device=device),
    )


def teacher_forced_predictions(model, traj, state_mean, state_std, max_ep_len, device):
    states = traj["states"]
    actions = traj["actions"]
    rtg = traj["returns_to_go"].reshape(-1, 1) / RETURN_SCALE

    T = states.shape[0]
    states_norm = (states - state_mean) / state_std

    states_t = torch.tensor(states_norm, dtype=torch.float32, device=device).unsqueeze(0)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)
    rtg_t = torch.tensor(rtg, dtype=torch.float32, device=device).unsqueeze(0)
    timesteps_t = torch.arange(T, device=device).clamp(max=max_ep_len - 1).unsqueeze(0)
    mask_t = torch.ones((1, T), dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(states_t, actions_t, rtg_t, timesteps_t, mask_t)
    return preds[0].cpu().numpy()


def run_single_seed(trajectories, seed):
    rng = np.random.RandomState(seed)
    shuffled_idx = rng.permutation(len(trajectories))

    n_val = min(NUM_HELDOUT_EPISODES, max(0, len(trajectories) - 1))
    val_idx = shuffled_idx[:n_val]
    train_idx = shuffled_idx[n_val:]

    train_trajectories = [trajectories[i] for i in train_idx]
    val_trajectories = [trajectories[i] for i in val_idx]

    state_dim = trajectories[0]["states"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    max_ep_len = max(t["states"].shape[0] for t in trajectories) + 10

    state_mean, state_std = compute_state_normalization(train_trajectories)

    model = DecisionTransformer(
        state_dim=state_dim, act_dim=act_dim, hidden_size=HIDDEN_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, max_ep_len=max_ep_len, dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for iteration in range(1, NUM_TRAIN_ITERS + 1):
        model.train()
        states, actions, rtg, timesteps, mask = get_batch(
            train_trajectories, BATCH_SIZE, CONTEXT_LENGTH, state_dim, act_dim,
            state_mean, state_std, max_ep_len, DEVICE,
        )
        action_preds = model(states, actions, rtg, timesteps, mask)
        loss_mask = mask.unsqueeze(-1)
        loss = (((action_preds - actions) * loss_mask) ** 2).sum() / loss_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
        optimizer.step()

    model.eval()
    real_all, pred_all = [], []
    for traj in val_trajectories:
        preds = teacher_forced_predictions(model, traj, state_mean, state_std, max_ep_len, DEVICE)
        real_all.append(traj["actions"])
        pred_all.append(preds)

    real_all = np.concatenate(real_all, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)

    magnitude = np.linalg.norm(real_all, axis=1)
    median_mag = np.median(magnitude)
    large_mask = magnitude > median_mag

    per_sample_sq_err = np.sum((pred_all - real_all) ** 2, axis=1)
    mse_large = per_sample_sq_err[large_mask].mean()
    baseline_large = np.sum(real_all[large_mask] ** 2, axis=1).mean()

    pct_improvement = 100 * (1 - mse_large / baseline_large) if baseline_large > 1e-9 else float("nan")

    return {
        "seed": seed,
        "val_files": [t["source_file"] for t in val_trajectories],
        "mse_large": mse_large,
        "baseline_large": baseline_large,
        "pct_improvement": pct_improvement,
    }


def main():
    print(f"Urzadzenie: {DEVICE}")
    trajectories = load_dataset(DATASET_FILE)
    print(f"Wczytano {len(trajectories)} trajektorii z {DATASET_FILE}")
    print(f"Testuje {len(SEEDS_TO_TEST)} roznych podzialow train/val, po {NUM_TRAIN_ITERS} iteracji kazdy...\n")

    results = []
    for seed in SEEDS_TO_TEST:
        print(f"--- Seed {seed} ---")
        r = run_single_seed(trajectories, seed)
        results.append(r)
        print(f"  Walidacja: {r['val_files']}")
        print(f"  DUZE akcje: MSE modelu={r['mse_large']:.6f}  baseline={r['baseline_large']:.6f}  "
              f"poprawa={r['pct_improvement']:+.1f}%\n")

    pct_values = [r["pct_improvement"] for r in results if not np.isnan(r["pct_improvement"])]

    print("=" * 60)
    print("PODSUMOWANIE (poprawa modelu vs baseline na DUZYCH akcjach):")
    for r in results:
        print(f"  seed={r['seed']:2d}: {r['pct_improvement']:+7.1f}%   (val: {r['val_files']})")

    print(f"\nSrednia poprawa: {np.mean(pct_values):+.1f}%")
    print(f"Odchylenie std:  {np.std(pct_values):.1f} punktow procentowych")
    print(f"Zakres: {min(pct_values):+.1f}% do {max(pct_values):+.1f}%")

    if np.std(pct_values) > 15:
        print(
            "\n-> Wysoka wariancja miedzy seedami sugeruje, ze wynik jest bardzo "
            "niestabilny/przypadkowy przy tej ilosci danych - potrzeba wiecej epizodow, "
            "zeby miec wiarygodna ocene jakosci modelu."
        )
    elif np.mean(pct_values) > 10:
        print("\n-> Model konsekwentnie bije baseline na duzych akcjach - dobry, stabilny sygnal.")
    else:
        print(
            "\n-> Model srednio ledwo bije (lub nie bije) baseline na duzych akcjach, "
            "niezaleznie od podzialu - to sugeruje ze problem NIE jest przypadkiem "
            "(zlym seedem), tylko systematycznym brakiem wystarczajacej ilosci/jakosci "
            "danych dla tego typu (duzych, rzadkich) akcji."
        )


if __name__ == "__main__":
    main()