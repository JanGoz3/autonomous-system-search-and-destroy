"""
Ocena DT: sweep po podzialach train/val + diagnostyka copycat.

Polaczenie dawnych seed_sweep.py i test_copycat.py - wspolne sa i tak wszystkie
funkcje pomocnicze (okna kontekstu, predykcja autoregresywna, blad katowy).

Uzycie:
    python seed.py              # sweep: trenuje N modeli, raportuje blad autoregresywny
    python seed.py copycat      # diagnostyka na istniejacym dt_checkpoint.pt

METRYKA: naglowkiem jest blad AUTOREGRESYWNY, nie teacher-forced. Teacher forcing
potrafi myslic za model: przy relabelingu po dystansie kolejne etykiety sa niemal
identyczne, wiec skopiowanie a_{t-1} podanego na wejsciu daje swietny wynik, ktory
w zamknietej petli nie istnieje. Autoregresja podaje modelowi jego WLASNE poprzednie
wyjscia - to najblizszy Pythonowi odpowiednik petli w DTInference.cs.
"""

import pickle
import sys

import numpy as np
import torch

import train_dt
from decision_transformer import DecisionTransformer

torch.backends.mha.set_fastpath_enabled(False)

CHECKPOINT_FILE = "dt_checkpoint.pt"
DATASET_FILE = "dt_dataset.pkl"

SEEDS_TO_TEST = [0, 1, 2, 3, 4]
NUM_TRAIN_ITERS = 1500
EVAL_BATCH = 128
DEVICE = "cpu"


# ===========================================================================
# Wspolne
# ===========================================================================

def load_dataset(path=DATASET_FILE):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data["trajectories"], data.get("action_scale", 1.0)
    return data, 1.0


def build_model(cfg):
    return DecisionTransformer(
        state_dim=cfg["state_dim"], act_dim=cfg["act_dim"],
        hidden_size=cfg["hidden_size"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], max_ep_len=cfg["max_ep_len"],
        action_head=cfg.get("action_head", "continuous"),
        n_dir_bins=cfg.get("n_dir_bins", 36),
    )


def model_from_ckpt(ckpt):
    model = build_model(ckpt["config"])
    model.load_state_dict({k: v.cpu() for k, v in ckpt["model_state_dict"].items()})
    model.eval()
    return model


def load_checkpoint(path=CHECKPOINT_FILE):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    return model_from_ckpt(ckpt), ckpt


def held_out(trajectories, ckpt):
    names = ckpt.get("held_out_files", [])
    return [t for t in trajectories
            if t.get("group", t["source_file"]) in names]


def prep(traj, ckpt):
    """Stan znormalizowany + return-to-go przeskalowany, tak jak w treningu."""
    cfg = ckpt["config"]
    states = traj["states"]
    if cfg.get("use_yaw_sincos"):
        yaw = np.radians(states[:, 2].astype(np.float64))
        states = np.column_stack([states[:, 0:2], np.sin(yaw), np.cos(yaw),
                                  states[:, 3:]]).astype(np.float32)
    s_norm = ((states - ckpt["state_mean"]) / ckpt["state_std"]).astype(np.float32)
    rtg = (traj["returns_to_go"].reshape(-1, 1) / cfg["return_scale"]).astype(np.float32)
    return s_norm, rtg


def build_windows(s_norm, actions, rtg, K, max_ep_len):
    """Okno K DOKLADNIE takie jak w treningu. Podanie modelowi calej trajektorii
    naraz mierzyloby go w rezimie, ktorego nigdy nie widzial."""
    T, sd, ad = len(s_norm), s_norm.shape[1], actions.shape[1]
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
        TS[i, K - n:] = np.clip(np.arange(lo, i + 1), 0, max_ep_len - 1)
        M[i, K - n:] = 1.0
    return S, A, R, TS, M


def predict_batched(model, S, A, R, TS, M):
    T = len(S)
    out = np.zeros((T, A.shape[2]), np.float32)
    with torch.no_grad():
        for b in range(0, T, EVAL_BATCH):
            sl = slice(b, min(b + EVAL_BATCH, T))
            y = model(torch.tensor(S[sl]), torch.tensor(A[sl]), torch.tensor(R[sl]),
                      torch.tensor(TS[sl]), torch.tensor(M[sl]))
            out[sl] = y[:, -1].numpy()
    return out


def predict_autoregressive(model, s_norm, rtg, ckpt, act_dim):
    """Kontekst z WLASNYCH predykcji modelu, stany prawdziwe."""
    cfg = ckpt["config"]
    K, max_ep_len = cfg["context_length"], cfg["max_ep_len"]
    T = len(s_norm)
    own = np.zeros((T, act_dim), np.float32)

    with torch.no_grad():
        for i in range(T):
            lo = max(0, i - K + 1)
            n = i - lo + 1
            S = np.zeros((1, K, s_norm.shape[1]), np.float32)
            A = np.zeros((1, K, act_dim), np.float32)
            R = np.zeros((1, K, 1), np.float32)
            TS = np.zeros((1, K), np.int64)
            M = np.zeros((1, K), np.float32)
            S[0, K - n:] = s_norm[lo:i + 1]
            A[0, K - n:] = own[lo:i + 1]
            R[0, K - n:] = rtg[lo:i + 1]
            TS[0, K - n:] = np.clip(np.arange(lo, i + 1), 0, max_ep_len - 1)
            M[0, K - n:] = 1.0
            y = model(torch.tensor(S), torch.tensor(A), torch.tensor(R),
                      torch.tensor(TS), torch.tensor(M))
            own[i] = y[0, -1].numpy()
    return own


def angular_error_deg(pred, true):
    pn, tn = np.linalg.norm(pred, axis=1), np.linalg.norm(true, axis=1)
    ok = (pn > 1e-6) & (tn > 1e-6)
    cos = np.clip((pred[ok] * true[ok]).sum(1) / (pn[ok] * tn[ok]), -1, 1)
    return np.degrees(np.arccos(cos))


def valid_of(traj):
    return traj["valid"] if "valid" in traj else np.ones(len(traj["actions"]), bool)


# ===========================================================================
# Sweep
# ===========================================================================

def run_single_seed(trajectories, seed):
    ckpt, best_val = train_dt.train_once(
        split_seed=seed, num_iters=NUM_TRAIN_ITERS, verbose=False, save_path=None)

    model = model_from_ckpt(ckpt)
    cfg = ckpt["config"]
    K = cfg["context_length"]

    A_tf, A_ar, Y, V = [], [], [], []
    for traj in held_out(trajectories, ckpt):
        s_norm, rtg = prep(traj, ckpt)
        acts = traj["actions"]
        S, A, R, TS, M = build_windows(s_norm, acts, rtg, K, cfg["max_ep_len"])
        # teacher forcing musi respektowac to, czym model byl karmiony w treningu
        A_in = np.zeros_like(A) if cfg.get("zero_actions") else A
        A_tf.append(predict_batched(model, S, A_in, R, TS, M))
        A_ar.append(predict_autoregressive(model, s_norm, rtg, ckpt, acts.shape[1]))
        Y.append(acts)
        V.append(valid_of(traj))

    valid = np.concatenate(V)
    y = np.concatenate(Y)[valid]
    tf = np.concatenate(A_tf)[valid]
    ar = np.concatenate(A_ar)[valid]

    return {
        "seed": seed,
        "best_val": best_val,
        "tf_median": float(np.median(angular_error_deg(tf, y))),
        "ar_median": float(np.median(angular_error_deg(ar, y))),
        "ar_under45": 100 * float((angular_error_deg(ar, y) < 45).mean()),
        "ar_len_ratio": float(np.linalg.norm(ar, axis=1).mean()
                              / max(np.linalg.norm(y, axis=1).mean(), 1e-9)),
    }


def run_sweep():
    # Trajektorie zostaja z SUROWYM yaw. Konwersje na sin/cos robi wylacznie
    # prep(), na podstawie flagi z checkpointu - i robi to na kopii, nie
    # mutujac datasetu. Konwertowanie takze tutaj rozszerzaloby stan dwa razy
    # (20 -> 21 -> 22) i prep() nie zgodzilby sie z wymiarem state_mean.
    trajectories, action_scale = train_dt.load_dataset(train_dt.DATASET_FILE)

    print(f"Urzadzenie: {train_dt.DEVICE}   trajektorii: {len(trajectories)}   "
          f"action_scale={action_scale:.4f} m")
    print(f"ZERO_ACTIONS_IN_CONTEXT = {train_dt.ZERO_ACTIONS_IN_CONTEXT}   "
          f"ACTION_HEAD = {train_dt.ACTION_HEAD}"
          + (f" ({train_dt.N_DIR_BINS} binow)" if train_dt.ACTION_HEAD == "discrete" else ""))
    print(f"{len(SEEDS_TO_TEST)} podzialow x {NUM_TRAIN_ITERS} iteracji\n")

    results = [run_single_seed(trajectories, s) for s in SEEDS_TO_TEST]
    for r in results:
        print(f"seed={r['seed']}  AUTOREGRESYWNIE mediana={r['ar_median']:5.1f} st  "
              f"<45st={r['ar_under45']:4.0f}%  dl/prawda={r['ar_len_ratio']:.2f}   "
              f"(teacher forcing: {r['tf_median']:5.1f} st)")

    ar = np.array([r["ar_median"] for r in results])
    tf = np.array([r["tf_median"] for r in results])
    ratio = np.array([r["ar_len_ratio"] for r in results])
    gap = ar - tf

    print("\n" + "=" * 70)
    print(f"Blad autoregresywny (mediana): {ar.mean():5.1f} st  +/- {ar.std():.1f}")
    print(f"Teacher forcing:               {tf.mean():5.1f} st  +/- {tf.std():.1f}")
    print(f"Luka TF -> autoregresja:       {gap.mean():+5.1f} st")
    print(f"Losowy kierunek:                90.0 st")
    print(f"Dlugosc predykcji / prawdy:     {ratio.mean():.2f}")

    print("\nOdczyt:")
    if ar.mean() > 70:
        print("  Blad bliski losowemu - model nie przewiduje kierunku ze stanu.")
    elif ar.mean() < 45 and gap.mean() < 10:
        print("  Model przewiduje kierunek ze stanu i nie rozjezdza sie w autoregresji.")
        print("  Mozna przechodzic do eksportu ONNX i testu zamknietej petli w Unity.")
    elif gap.mean() > 20:
        print("  Duza luka TF -> autoregresja: model wykorzystuje historie akcji")
        print("  jako skrot. Sprawdz ZERO_ACTIONS_IN_CONTEXT i decymacje danych.")
    else:
        print("  Posrednio - model cos umie ze stanu, ale daleko mu do etykiet.")

    if ratio.mean() < 0.6:
        print(f"\n  Predykcje {ratio.mean():.2f}x krotsze od prawdziwych akcji -")
        print("  podpis regresji L2 na rozkladzie wielomodalnym. Wiecej danych tego")
        print("  nie naprawi; potrzebna glowica dyskretna zamiast MSE.")


# ===========================================================================
# Copycat
# ===========================================================================

def run_copycat():
    """Czy model czyta stan, czy calkuje wlasne poprzednie wyjscie.

      teacher          - kontekst z PRAWDZIWYMI poprzednimi akcjami
      bez akcji        - akcje w kontekscie wyzerowane
      autoregresywnie  - kontekst z WLASNYMI predykcjami modelu
    """
    model, ckpt = load_checkpoint()
    trajectories, action_scale = load_dataset()
    cfg = ckpt["config"]
    K = cfg["context_length"]

    val = held_out(trajectories, ckpt)
    if not val:
        print("Brak epizodow walidacyjnych - przerwane.")
        return
    print(f"Epizody: {[t['source_file'] for t in val]}")
    print(f"context_length={K}, action_scale={action_scale:.4f} m\n")

    res = {"teacher": [], "bez akcji": [], "autoregresywnie": []}
    truth, valid_all, prev_all = [], [], []

    for traj in val:
        s_norm, rtg = prep(traj, ckpt)
        acts = traj["actions"]
        S, A, R, TS, M = build_windows(s_norm, acts, rtg, K, cfg["max_ep_len"])

        res["teacher"].append(predict_batched(model, S, A, R, TS, M))
        res["bez akcji"].append(predict_batched(model, S, np.zeros_like(A), R, TS, M))
        res["autoregresywnie"].append(
            predict_autoregressive(model, s_norm, rtg, ckpt, acts.shape[1]))

        truth.append(acts)
        prev_all.append(np.vstack([acts[:1], acts[:-1]]))
        valid_all.append(valid_of(traj))

    valid = np.concatenate(valid_all)
    y = np.concatenate(truth)[valid]
    prev = np.concatenate(prev_all)[valid]

    print(f"{'wariant':22s} {'mediana':>9s} {'srednia':>9s} {'<45st':>7s} {'dl/prawda':>10s}")
    print("-" * 62)
    meds = {}
    for name in ("teacher", "bez akcji", "autoregresywnie"):
        p = np.concatenate(res[name])[valid]
        ang = angular_error_deg(p, y)
        meds[name] = float(np.median(ang))
        ratio = np.linalg.norm(p, axis=1).mean() / np.linalg.norm(y, axis=1).mean()
        print(f"{name:22s} {np.median(ang):8.1f}st {ang.mean():8.1f}st "
              f"{100 * (ang < 45).mean():6.0f}% {ratio:10.2f}")

    ang_prev = angular_error_deg(prev, y)
    print(f"{'(kopiuj a_t-1)':22s} {np.median(ang_prev):8.1f}st {ang_prev.mean():8.1f}st "
          f"{100 * (ang_prev < 45).mean():6.0f}%")

    d_noact = meds["bez akcji"] - meds["teacher"]
    d_auto = meds["autoregresywnie"] - meds["teacher"]
    print("\nWnioski:")
    print(f"  koszt usuniecia poprzednich akcji: {d_noact:+.1f} st")
    print(f"  koszt autoregresji:                {d_auto:+.1f} st")

    if d_noact < 3 and d_auto < 3:
        print("\n  Model czyta stan, nie kopiuje poprzedniej akcji.")
    elif d_auto > 15:
        print("\n  Blad kumuluje sie autoregresywnie - w Unity model bedzie dryfowal.")
        print("  Naprawa: trenuj z wyzerowanymi akcjami w kontekscie.")
    else:
        print("\n  Czesciowa zaleznosc od poprzedniej akcji.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "copycat":
        run_copycat()
    else:
        run_sweep()