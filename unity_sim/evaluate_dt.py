import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decision_transformer import DecisionTransformer

torch.backends.mha.set_fastpath_enabled(False)


CHECKPOINT_FILE = "dt_checkpoint.pt"
DATASET_FILE = "dt_dataset.pkl"
PLOT_OUTPUT = "dt_evaluation.png"

MAX_ARENA_SIZE = 20.0
NUM_ARROWS = 25         # ile strzalek (predykcji) narysowac na wykresie

DEVICE = "cpu"


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = DecisionTransformer(
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
        hidden_size=cfg["hidden_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        max_ep_len=cfg["max_ep_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def local_to_world(local_x, local_z, yaw_deg):
    yaw_rad = np.radians(yaw_deg)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    world_dx = local_x * cos_y - local_z * sin_y
    world_dz = local_x * sin_y + local_z * cos_y
    return world_dx, world_dz


def teacher_forced_predictions(model, traj, state_mean, state_std, return_scale, max_ep_len, device):
    states = traj["states"]
    actions = traj["actions"]
    rtg = traj["returns_to_go"].reshape(-1, 1) / return_scale

    T = states.shape[0]
    states_norm = (states - state_mean) / state_std

    states_t = torch.tensor(states_norm, dtype=torch.float32, device=device).unsqueeze(0)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)
    rtg_t = torch.tensor(rtg, dtype=torch.float32, device=device).unsqueeze(0)
    timesteps_t = torch.arange(T, device=device).clamp(max=max_ep_len - 1).unsqueeze(0)
    mask_t = torch.ones((1, T), dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(states_t, actions_t, rtg_t, timesteps_t, mask_t)

    return preds[0].numpy()  # (T, act_dim)


def main():
    model, ckpt = load_checkpoint(CHECKPOINT_FILE)
    cfg = ckpt["config"]
    state_mean, state_std = ckpt["state_mean"], ckpt["state_std"]
    held_out_files = ckpt.get("held_out_files", [])

    with open(DATASET_FILE, "rb") as f:
        trajectories = pickle.load(f)

    val_trajectories = [t for t in trajectories if t["source_file"] in held_out_files]

    if len(val_trajectories) == 0:
        print("UWAGA: brak epizodow walidacyjnych w datasecie (held_out_files puste "
              "lub zaden plik sie nie zgadza) - ocena bedzie na epizodach TRENINGOWYCH, "
              "co nie mowi nic o generalizacji, tylko o tym czy model w ogole cokolwiek sie nauczyl.")
        val_trajectories = trajectories

    print(f"Epizody uzyte do oceny: {[t['source_file'] for t in val_trajectories]}")

    all_mse = []
    all_real_actions = []
    all_pred_actions = []

    for traj in val_trajectories:
        preds = teacher_forced_predictions(
            model, traj, state_mean, state_std, cfg["return_scale"], cfg["max_ep_len"], DEVICE
        )
        mse = np.mean((preds - traj["actions"]) ** 2)
        all_mse.append(mse)
        all_real_actions.append(traj["actions"])
        all_pred_actions.append(preds)
        print(f"  {traj['source_file']}: MSE = {mse:.6f} ({traj['states'].shape[0]} krokow)")

    print(f"\nSrednie MSE na zbiorze oceny: {np.mean(all_mse):.6f}")
    print(
        "Punkt odniesienia: akcje sa znormalizowane (Delta pozycji / "
        f"{MAX_ARENA_SIZE}), wiec MSE rzedu 0.001-0.01 sugeruje sensowne dopasowanie; "
        "MSE zblizone do wariancji samych akcji oznacza ze model NIE nauczyl sie nic "
        "ponad przewidywanie sredniej."
    )

    baseline_mse = np.mean([
        np.mean(traj["actions"] ** 2) for traj in val_trajectories
    ])
    print(f"Baseline (przewidywanie zawsze 'stoj w miejscu', akcja=0): MSE = {baseline_mse:.6f}")
    if np.mean(all_mse) < baseline_mse:
        print("-> Model bije naiwny baseline - dobry znak, uczy sie czegos ponad staly wzorzec.")
    else:
        print("-> Model NIE bije naiwnego baseline - to zly znak, prawdopodobnie za malo danych "
              "lub trzeba dluzej trenowac / dostroic hiperparametry.")

    real_all = np.concatenate(all_real_actions, axis=0)      # (N, 2)
    pred_all = np.concatenate(all_pred_actions, axis=0)      # (N, 2)

    real_magnitude = np.linalg.norm(real_all, axis=1)
    median_mag = np.median(real_magnitude)

    small_mask = real_magnitude <= median_mag
    large_mask = ~small_mask

    per_sample_sq_err = np.sum((pred_all - real_all) ** 2, axis=1)

    mse_small = per_sample_sq_err[small_mask].mean()
    mse_large = per_sample_sq_err[large_mask].mean()

    baseline_small = np.sum(real_all[small_mask] ** 2, axis=1).mean()
    baseline_large = np.sum(real_all[large_mask] ** 2, axis=1).mean()

    print("\n--- Podzial wedlug wielkosci rzeczywistej akcji (mediana = "
          f"{median_mag:.4f}) ---")
    print(f"MALE akcje  (<= mediana, N={small_mask.sum():5d}): "
          f"MSE modelu={mse_small:.6f}  |  baseline='stoj'={baseline_small:.6f}  |  "
          f"model {'bije' if mse_small < baseline_small else 'NIE bije'} baseline "
          f"({100*(1 - mse_small/baseline_small):+.1f}%)")
    print(f"DUZE  akcje (>  mediana, N={large_mask.sum():5d}): "
          f"MSE modelu={mse_large:.6f}  |  baseline='stoj'={baseline_large:.6f}  |  "
          f"model {'bije' if mse_large < baseline_large else 'NIE bije'} baseline "
          f"({100*(1 - mse_large/baseline_large):+.1f}%)")
    print(
        "\nInterpretacja: jesli model wyraznie bije baseline na MALYCH akcjach, ale "
        "NIE bije (albo ledwo) na DUZYCH - to znaczy ze model dobrze nauczyl sie "
        "drobnych, lokalnych ruchow, ale nie radzi sobie z rzadszymi, szybkimi/dlugimi "
        "'doskokami' w danych (typowe przy niezbalansowanym rozkladzie wielkosci akcji)."
    )

    traj = val_trajectories[0]
    preds = teacher_forced_predictions(
        model, traj, state_mean, state_std, cfg["return_scale"], cfg["max_ep_len"], DEVICE
    )

    pos = traj["states"][:, 0:2]  # posX, posZ (pierwsze 2 kolumny wektora stanu)
    yaw = traj["states"][:, 2]    # yaw

    T = pos.shape[0]
    sample_idx = np.linspace(0, T - 1, min(NUM_ARROWS, T), dtype=int)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(pos[:, 0], pos[:, 1], color="gray", linewidth=1, label="Rzeczywista trasa (pozycja auta)")
    ax.scatter(pos[0, 0], pos[0, 1], color="green", s=100, zorder=5, label="Start")
    ax.scatter(pos[-1, 0], pos[-1, 1], color="black", s=100, zorder=5, label="Koniec")

    arrow_scale = MAX_ARENA_SIZE  # odwraca normalizacje /MAX_ARENA_SIZE z build_dataset.py

    for i in sample_idx:
        real_dx, real_dz = local_to_world(
            traj["actions"][i, 0] * arrow_scale, traj["actions"][i, 1] * arrow_scale, yaw[i]
        )
        pred_dx, pred_dz = local_to_world(
            preds[i, 0] * arrow_scale, preds[i, 1] * arrow_scale, yaw[i]
        )

        ax.arrow(pos[i, 0], pos[i, 1], real_dx, real_dz,
                 color="blue", width=0.02, head_width=0.15, alpha=0.7,
                 length_includes_head=True,
                 label="Rzeczywisty waypoint" if i == sample_idx[0] else None)
        ax.arrow(pos[i, 0], pos[i, 1], pred_dx, pred_dz,
                 color="red", width=0.02, head_width=0.15, alpha=0.7,
                 length_includes_head=True,
                 label="Przewidziany waypoint (model)" if i == sample_idx[0] else None)

    ax.set_xlabel("posX")
    ax.set_ylabel("posZ")
    ax.set_title(f"Ocena DT na epizodzie: {traj['source_file']}\n"
                 f"Niebieskie strzalki = rzeczywistosc, Czerwone = predykcja modelu")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.savefig(PLOT_OUTPUT, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres do {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()