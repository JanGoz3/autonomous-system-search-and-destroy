import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.DecisionTransformer.decision_transformer import DecisionTransformer

torch.backends.mha.set_fastpath_enabled(False)

CHECKPOINT_FILE = "dt_checkpoint.pt"
DATASET_FILE = "dt_dataset.pkl"
PLOT_OUTPUT = "dt_evaluation.png"

NUM_ARROWS = 25
EVAL_BATCH = 128
DEVICE = "cpu"


def local_to_world(local_x, local_z, yaw_deg):
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    return local_x * c + local_z * s, -local_x * s + local_z * c


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = DecisionTransformer(
        state_dim=cfg["state_dim"], act_dim=cfg["act_dim"],
        hidden_size=cfg["hidden_size"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], max_ep_len=cfg["max_ep_len"],
        action_head=cfg.get("action_head", "continuous"),
        n_dir_bins=cfg.get("n_dir_bins", 36),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data["trajectories"], data.get("action_scale", 1.0)
    return data, 1.0


def apply_yaw_sincos(traj_states):
    yaw = np.radians(traj_states[:, 2].astype(np.float64))
    return np.column_stack([traj_states[:, 0:2],
                            np.sin(yaw), np.cos(yaw),
                            traj_states[:, 3:]]).astype(np.float32)


def windowed_predictions(model, traj, ckpt, device=DEVICE):
    cfg = ckpt["config"]
    K = cfg["context_length"]
    states = traj["states"]
    if cfg.get("use_yaw_sincos"):
        states = apply_yaw_sincos(states)

    s_norm = (states - ckpt["state_mean"]) / ckpt["state_std"]
    actions = traj["actions"]
    rtg = traj["returns_to_go"].reshape(-1, 1) / cfg["return_scale"]
    T, sd, ad = len(states), s_norm.shape[1], actions.shape[1]

    S = np.zeros((T, K, sd), np.float32)
    A = np.zeros((T, K, ad), np.float32)
    R = np.zeros((T, K, 1), np.float32)
    TS = np.zeros((T, K), np.int64)
    M = np.zeros((T, K), np.float32)

    for i in range(T):
        lo = max(0, i - K + 1)
        n = i - lo + 1
        S[i, K - n:] = s_norm[lo:i + 1]
        A[i, K - n:] = actions[lo:i + 1]
        R[i, K - n:] = rtg[lo:i + 1]
        TS[i, K - n:] = np.clip(np.arange(lo, i + 1), 0, cfg["max_ep_len"] - 1)
        M[i, K - n:] = 1.0

    preds = np.zeros((T, ad), np.float32)
    with torch.no_grad():
        for b in range(0, T, EVAL_BATCH):
            sl = slice(b, min(b + EVAL_BATCH, T))
            out = model(torch.tensor(S[sl]), torch.tensor(A[sl]),
                        torch.tensor(R[sl]), torch.tensor(TS[sl]),
                        torch.tensor(M[sl]))
            preds[sl] = out[:, -1].numpy()
    return preds


def angular_error_deg(pred, true):
    pn = np.linalg.norm(pred, axis=1)
    tn = np.linalg.norm(true, axis=1)
    ok = (pn > 1e-6) & (tn > 1e-6)
    cos = np.clip((pred[ok] * true[ok]).sum(1) / (pn[ok] * tn[ok]), -1, 1)
    return np.degrees(np.arccos(cos)), ok


def main():
    model, ckpt = load_checkpoint(CHECKPOINT_FILE)
    cfg = ckpt["config"]
    action_scale = cfg.get("action_scale", 1.0)
    trajectories, _ = load_dataset(DATASET_FILE)

    held = ckpt.get("held_out_files", [])
    val = [t for t in trajectories if t.get("group", t["source_file"]) in held]
    if not val:
        print("UWAGA: brak epizodow walidacyjnych - oceniam na treningowych, "
              "co nie mowi nic o generalizacji.")
        val = trajectories

    print(f"Epizody oceny: {[t['source_file'] for t in val]}")
    print(f"action_scale = {action_scale:.4f} m/jednostke\n")

    P, Y, V = [], [], []
    for traj in val:
        P.append(windowed_predictions(model, traj, ckpt))
        Y.append(traj["actions"])
        V.append(traj["valid"] if "valid" in traj
                 else np.ones(len(traj["actions"]), bool))

    pred = np.concatenate(P)
    true = np.concatenate(Y)
    valid = np.concatenate(V)
    p, y = pred[valid], true[valid]
    print(f"Probek ocenianych (tylko valid): {len(y)} z {len(true)}")

    # --- baseline'y ---
    # "Stoj w miejscu" stracil sens: przy relabelingu po dystansie akcja nigdy
    # nie jest zerowa, wiec ten baseline jest trywialnie do pobicia i nic nie mowi.
    prev = np.concatenate([np.vstack([tr["actions"][:1], tr["actions"][:-1]])
                           for tr in val])[valid]
    baselines = {
        "zero (stoj)": np.zeros_like(y),
        "srednia akcja treningowa": np.tile(y.mean(0), (len(y), 1)),
        "prosto do przodu": np.tile([0.0, np.linalg.norm(y, axis=1).mean()], (len(y), 1)),
        "powtorz poprzednia akcje": prev,
    }

    mse_model = np.mean((p - y) ** 2)
    print(f"MSE modelu: {mse_model:.6f}\n")
    print(f"{'baseline':30s} {'MSE':>10s} {'poprawa':>10s}")
    print("-" * 52)
    for name, b in baselines.items():
        mb = np.mean((b - y) ** 2)
        print(f"{name:30s} {mb:10.6f} {100 * (1 - mse_model / mb):+9.1f}%")

    # --- blad katowy ---
    # Dlugosc akcji jest teraz niemal stala (waypoint w stalej odleglosci),
    # wiec o jakosci decyduje KIERUNEK, nie wielkosc.
    ang, ok = angular_error_deg(p, y)
    ang_prev, ok_p = angular_error_deg(prev, y)
    print(f"\nBlad kierunku (stopnie, N={ok.sum()}):")
    print(f"  model:                    mediana={np.median(ang):5.1f}  "
          f"srednia={ang.mean():5.1f}  <45st: {100 * (ang < 45).mean():.0f}%")
    print(f"  'powtorz poprzednia':     mediana={np.median(ang_prev):5.1f}  "
          f"srednia={ang_prev.mean():5.1f}  <45st: {100 * (ang_prev < 45).mean():.0f}%")
    print("  Losowy kierunek dalby mediane 90 st.")

    print(f"\nDlugosc akcji [m]: rzeczywista={np.linalg.norm(y, axis=1).mean() * action_scale:.2f}, "
          f"predykcja={np.linalg.norm(p, axis=1).mean() * action_scale:.2f}")
    print("  Predykcja wyraznie krotsza od rzeczywistej = model usrednia "
          "wielomodalny rozklad kierunkow i wypuszcza wektor do srodka.")

    # --- wykres ---
    traj = val[0]
    preds = windowed_predictions(model, traj, ckpt)
    pos, yaw = traj["states"][:, 0:2], traj["states"][:, 2]
    vmask = traj["valid"] if "valid" in traj else np.ones(len(pos), bool)
    idxs = np.flatnonzero(vmask)
    idxs = idxs[np.linspace(0, len(idxs) - 1, min(NUM_ARROWS, len(idxs))).astype(int)]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(pos[:, 0], pos[:, 1], color="gray", lw=1, label="Rzeczywista trasa")
    ax.scatter(*pos[0], color="green", s=100, zorder=5, label="Start")
    ax.scatter(*pos[-1], color="black", s=100, zorder=5, label="Koniec")

    for n, i in enumerate(idxs):
        for arr, col, lab in ((traj["actions"][i], "blue", "Rzeczywisty waypoint"),
                              (preds[i], "red", "Predykcja modelu")):
            dx, dz = local_to_world(arr[0] * action_scale, arr[1] * action_scale, yaw[i])
            ax.arrow(pos[i, 0], pos[i, 1], dx, dz, color=col, width=0.02,
                     head_width=0.15, alpha=0.7, length_includes_head=True,
                     label=lab if n == 0 else None)

    ax.set_xlabel("posX")
    ax.set_ylabel("posZ")
    ax.set_title(f"Ocena DT: {traj['source_file']}\nniebieskie = prawda, czerwone = model")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig(PLOT_OUTPUT, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres do {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()