import numpy as np
import torch

import train_dt
from decision_transformer import DecisionTransformer
from test_copycat import (predict_autoregressive, build_windows,
                          predict_batched, angular_error_deg, prep)

SEEDS_TO_TEST = [0, 1, 2, 3, 4]
NUM_TRAIN_ITERS = 1500


def model_from_ckpt(ckpt):
    cfg = ckpt["config"]
    model = DecisionTransformer(
        state_dim=cfg["state_dim"], act_dim=cfg["act_dim"],
        hidden_size=cfg["hidden_size"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], max_ep_len=cfg["max_ep_len"],
        action_head=cfg.get("action_head", "continuous"),
        n_dir_bins=cfg.get("n_dir_bins", 36),
    )
    model.load_state_dict({k: v.cpu() for k, v in ckpt["model_state_dict"].items()})
    model.eval()
    return model


def run_single_seed(trajectories, seed):
    ckpt, best_val = train_dt.train_once(
        split_seed=seed, num_iters=NUM_TRAIN_ITERS, verbose=False, save_path=None)

    model = model_from_ckpt(ckpt)
    cfg = ckpt["config"]
    K = cfg["context_length"]
    val = [t for t in trajectories if t.get("group", t["source_file"]) in ckpt["held_out_files"]]

    A_tf, A_ar, Y, V = [], [], [], []
    for traj in val:
        s_norm, rtg = prep(traj, ckpt)
        acts = traj["actions"]
        S, A, R, TS, M = build_windows(s_norm, acts, rtg, K, cfg["max_ep_len"])
        # teacher forcing musi respektowac to, czym model byl karmiony w treningu
        A_in = np.zeros_like(A) if cfg.get("zero_actions") else A
        A_tf.append(predict_batched(model, S, A_in, R, TS, M))
        A_ar.append(predict_autoregressive(model, s_norm, rtg, ckpt, acts.shape[1]))
        Y.append(acts)
        V.append(traj["valid"] if "valid" in traj else np.ones(len(acts), bool))

    valid = np.concatenate(V)
    y = np.concatenate(Y)[valid]
    tf = np.concatenate(A_tf)[valid]
    ar = np.concatenate(A_ar)[valid]

    ang_tf = angular_error_deg(tf, y)
    ang_ar = angular_error_deg(ar, y)
    return {
        "seed": seed,
        "best_val": best_val,
        "tf_median": float(np.median(ang_tf)),
        "ar_median": float(np.median(ang_ar)),
        "ar_under45": 100 * float((ang_ar < 45).mean()),
        "ar_len_ratio": float(np.linalg.norm(ar, axis=1).mean()
                              / max(np.linalg.norm(y, axis=1).mean(), 1e-9)),
    }


def main():
    trajectories, action_scale = train_dt.load_dataset(train_dt.DATASET_FILE)
    if train_dt.USE_YAW_SINCOS:
        train_dt.apply_yaw_sincos(trajectories)

    print(f"Urzadzenie: {train_dt.DEVICE}   trajektorii: {len(trajectories)}   "
          f"action_scale={action_scale:.4f} m")
    print(f"ZERO_ACTIONS_IN_CONTEXT = {train_dt.ZERO_ACTIONS_IN_CONTEXT}   "
          f"ACTION_HEAD = {train_dt.ACTION_HEAD}"
          + (f" ({train_dt.N_DIR_BINS} binow)" if train_dt.ACTION_HEAD == "discrete" else ""))
    print(f"{len(SEEDS_TO_TEST)} podzialow x {NUM_TRAIN_ITERS} iteracji\n")

    results = []
    for seed in SEEDS_TO_TEST:
        r = run_single_seed(trajectories, seed)
        results.append(r)
        print(f"seed={seed}  AUTOREGRESYWNIE mediana={r['ar_median']:5.1f} st  "
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
        print("  Duza luka miedzy teacher forcing a autoregresja: model nadal")
        print("  wykorzystuje historie akcji jako skrot. Sprawdz, czy")
        print("  ZERO_ACTIONS_IN_CONTEXT jest wlaczone, i rozwaz decymacje danych")
        print("  do czestotliwosci decyzji z DTInference.")
    else:
        print("  Posrednio - model cos umie ze stanu, ale daleko mu do etykiet.")
        print("  Kolejny krok: decymacja do czestotliwosci decyzji, potem dane.")

    if ratio.mean() < 0.6:
        print(f"\n  Predykcje {ratio.mean():.2f}x krotsze od prawdziwych akcji -")
        print("  podpis regresji L2 na rozkladzie wielomodalnym. Wiecej danych tego")
        print("  nie naprawi; potrzebna glowica rozkladowa (dyskretyzacja kierunku")
        print("  + cross-entropy) zamiast MSE.")


if __name__ == "__main__":
    main()