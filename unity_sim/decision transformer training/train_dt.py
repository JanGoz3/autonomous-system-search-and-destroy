"""
Trening WaypointTransformera (behavior cloning, droga A).

Rozne wzgledem train_dt.py:
  * brak return_scale i returns_to_go
  * brak ZERO_ACTIONS_IN_CONTEXT - akcje nie wchodza do kontekstu w ogole
  * OKNO LOSOWANE PO KONCU, nie po poczatku. train_dt.py losowal si i bral
    states[si:si+K], wiec skrocone okna wypadaly na KONCU epizodu. W inferencji
    skrocone okno wystepuje na POCZATKU przebiegu. Tutaj losujemy indeks
    "teraz" i cofamy sie o K-1 krokow, co odtwarza sytuacje z inferencji.
  * metryki w stopniach, nie tylko wartosc lossu
"""

import pickle

import numpy as np
import torch
import torch.nn as nn

from models.DecisionTransformer.decision_transformer import WaypointTransformer

DATASET_FILE = "bc_dataset_pos0_scan1p_yolo2_dir.pkl"
CHECKPOINT_FILE = "bc_ckpt_dir.pt"

CONTEXT_LENGTH = 20

# 610 tys. parametrow na ~2200 niezaleznych decyzji (31 epizodow x ~70) dawalo
# najlepsza walidacje po 200 iteracjach z 8000 - model zapamietywal zbior szybciej
# niz robil jedno przejscie. KEEP_ALL_PHASES tworzy 10 kopii kazdego epizodu
# przesunietych o 0.1 s, ktore sa niemal duplikatami, wiec nominalne 310 sekwencji
# nie jest 310 niezaleznymi probkami.
HIDDEN_SIZE = 64
N_LAYER = 2
N_HEAD = 4
DROPOUT = 0.2

# Przyciecie znormalizowanego stanu. Na zebranych danych telem_8 (zyroskop Y) mial
# max |z| = 70.5 - jedno zdarzenie o ogromnej predkosci katowej po kolizji
# dominowalo normalizacje calej kolumny. MUSI byc identyczne w export_bc_to_onnx.py.
STATE_CLIP = 5.0

# Zakrety 45-70 deg to 7.5% probek i najgorszy wynik (32.6 deg bledu, 14% trafien).
# Ten udzial okien zakotwiczonych na kroku z |kat celu| >= TURN_MIN_DEG.
TURN_OVERSAMPLE = 0.35
TURN_MIN_DEG = 25.0

N_DIR_BINS = 36
MAG_WEIGHT = 1.0
LABEL_SMOOTH = 0.15

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
GRAD_NORM_CLIP = 1.0
WARMUP_ITERS = 200
NUM_TRAIN_ITERS = 10000

LOG_EVERY = 200
VAL_EVERY = 200
VAL_BATCHES = 16
EARLY_STOP_PATIENCE = 15       # w jednostkach VAL_EVERY

NUM_HELDOUT_GROUPS = 4
SPLIT_SEED = 42

USE_YAW_SINCOS = True

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def load_dataset(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["trajectories"], d["action_scale"], d["state_columns"]


def apply_yaw_sincos(trajectories, yaw_index):
    """yaw w stopniach -> (sin, cos). Musi byc odtworzone identycznie w grafie
    ONNX, bo Unity podaje surowy yaw."""
    for t in trajectories:
        s = t["states"]
        yaw = np.radians(s[:, yaw_index].astype(np.float64))
        t["states"] = np.column_stack([
            s[:, :yaw_index],
            np.sin(yaw).astype(np.float32),
            np.cos(yaw).astype(np.float32),
            s[:, yaw_index + 1:],
        ]).astype(np.float32)


def mark_turn_indices(trajectories, min_deg=TURN_MIN_DEG):
    """Dla kazdej trajektorii lista indeksow, w ktorych etykieta jest zakretem."""
    for t in trajectories:
        a = np.degrees(np.arctan2(t["actions"][:, 0], t["actions"][:, 1]))
        t["turn_idx"] = np.flatnonzero((np.abs(a) >= min_deg) & t["valid"])


def compute_state_normalization(trajectories):
    S = np.concatenate([t["states"] for t in trajectories], axis=0)
    return S.mean(0).astype(np.float32), (S.std(0) + 1e-6).astype(np.float32)


def get_batch(trajectories, batch_size, K, state_dim, act_dim,
              state_mean, state_std, device, rng):
    lengths = np.array([t["states"].shape[0] for t in trajectories], dtype=np.float64)
    inds = rng.choice(len(trajectories), size=batch_size, p=lengths / lengths.sum())

    s_l, a_l, m_l, v_l = [], [], [], []
    for idx in inds:
        traj = trajectories[idx]
        n = traj["states"].shape[0]

        # z prawdopodobienstwem TURN_OVERSAMPLE zakotwicz okno na zakrecie
        ti = traj.get("turn_idx")
        if ti is not None and len(ti) and rng.rand() < TURN_OVERSAMPLE:
            te = int(ti[rng.randint(0, len(ti))])
        else:
            te = rng.randint(0, n)             # indeks "teraz"
        ts = max(0, te - K + 1)

        s = (traj["states"][ts:te + 1] - state_mean) / state_std
        s = np.clip(s, -STATE_CLIP, STATE_CLIP)
        a = traj["actions"][ts:te + 1]
        v = traj["valid"][ts:te + 1].astype(np.float32)

        tlen = s.shape[0]
        pad = K - tlen
        z = lambda shape: np.zeros(shape, dtype=np.float32)

        s_l.append(np.concatenate([z((pad, state_dim)), s]))
        a_l.append(np.concatenate([z((pad, act_dim)), a]))
        m_l.append(np.concatenate([z((pad,)), np.ones(tlen, dtype=np.float32)]))
        v_l.append(np.concatenate([z((pad,)), v]))

    to = lambda arr: torch.tensor(np.stack(arr), dtype=torch.float32, device=device)
    return to(s_l), to(a_l), to(m_l), to(v_l)


@torch.no_grad()
def angular_metrics(model, s, m, a, lm, n_bins):
    """Sredni i medianowy blad kierunku w stopniach + trafienia w kubelek."""
    logits, mag = model.heads(s, m)
    centers = model.bin_centers
    pred_ang = centers[logits.argmax(dim=-1)]
    tgt_ang = torch.atan2(a[..., 0], a[..., 1])

    d = torch.atan2(torch.sin(pred_ang - tgt_ang), torch.cos(pred_ang - tgt_ang))
    d = d.abs() * 180.0 / np.pi

    sel = lm > 0
    if sel.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    dv = d[sel]
    mag_err = (mag[sel] - a[..., :][sel].norm(dim=-1)).abs()
    return (dv.mean().item(), dv.median().item(),
            (dv <= 15.0).float().mean().item(), mag_err.mean().item())


@torch.no_grad()
def report_by_turn_bucket(model, trajectories, args, rng, action_scale, n_batches=40):
    """Rozbicie bledu po WIELKOSCI zakretu.

    Zagregowana celnosc jest zdominowana przez prosta - linia bazowa 'zawsze
    prosto' trafia w wiekszosc probek. O przejechaniu okrazenia decyduja
    wylacznie zakrety, wiec to jest metryka, na ktora trzeba patrzec.
    """
    model.eval()
    tgt_all, err_all = [], []
    for _ in range(n_batches):
        s, a, m, v = get_batch(trajectories, *args, rng)
        lm = (m * v) > 0
        logits, _ = model.heads(s, m)
        pred = model.bin_centers[logits.argmax(dim=-1)]
        tgt = torch.atan2(a[..., 0], a[..., 1])
        d = torch.atan2(torch.sin(pred - tgt), torch.cos(pred - tgt)).abs() * 180 / np.pi
        tgt_all.append((tgt[lm].abs() * 180 / np.pi).cpu().numpy())
        err_all.append(d[lm].cpu().numpy())

    t = np.concatenate(tgt_all)
    e = np.concatenate(err_all)
    edges = [(0, 10), (10, 25), (25, 45), (45, 70)]

    print("\n--- Blad kierunku wzgledem wielkosci zakretu (walidacja) ---")
    print(f"{'|kat celu|':>14} {'probek':>8} {'udzial':>7} {'sr. blad':>10} "
          f"{'med.':>7} {'<=15 deg':>9} {'baza':>7}")
    for lo, hi in edges:
        sel = (t >= lo) & (t < hi)
        if sel.sum() == 0:
            continue
        # linia bazowa 'zawsze prosto' = blad rowny samemu katowi celu
        print(f"{lo:5.0f}-{hi:<3.0f} deg {sel.sum():8d} {100*sel.mean():6.1f}% "
              f"{e[sel].mean():9.1f} deg {np.median(e[sel]):6.1f} "
              f"{100*(e[sel] <= 15).mean():8.0f}% {t[sel].mean():6.1f}")
    print("  kolumna 'baza' = sredni blad polityki 'zawsze prosto' w tym kubelku")
    turns = t >= 25
    if turns.sum():
        print(f"\n  NA ZAKRETACH (>=25 deg): sredni blad {e[turns].mean():.1f} deg, "
              f"trafien <=15 deg {100*(e[turns] <= 15).mean():.0f}% "
              f"(baza: {t[turns].mean():.1f} deg, 0%)")


def main():
    trajectories, action_scale, state_columns = load_dataset(DATASET_FILE)
    yaw_index = state_columns.index("yaw")
    if USE_YAW_SINCOS:
        apply_yaw_sincos(trajectories, yaw_index)

    groups = sorted({t["group"] for t in trajectories})
    rng_split = np.random.RandomState(SPLIT_SEED)
    n_val = min(NUM_HELDOUT_GROUPS, max(1, len(groups) // 5))

    # Podzial STRATYFIKOWANY po rodzaju epizodu. Losowanie bez tego wybieralo same
    # grupy 'clean', przez co walidacja mierzyla wylacznie jazde po idealnej linii,
    # a nie odzyskiwanie po zjechaniu z niej - i dlatego strata walidacyjna
    # wychodzila NIZSZA od treningowej. EpisodeDirector koduje warunek w nazwie
    # (bc_fwd_noisy_r003), wiec da sie to rozdzielic po prostym dopasowaniu.
    noisy = [g for g in groups if "noisy" in g]
    clean = [g for g in groups if "noisy" not in g]

    held = set()
    if noisy and clean:
        n_noisy = max(1, n_val // 2)
        n_clean = max(1, n_val - n_noisy)
        held |= {noisy[i] for i in rng_split.permutation(len(noisy))[:n_noisy]}
        held |= {clean[i] for i in rng_split.permutation(len(clean))[:n_clean]}
        print(f"Podzial stratyfikowany: {n_noisy} grup z szumem + {n_clean} czystych")
    else:
        perm = rng_split.permutation(len(groups))
        held = {groups[i] for i in perm[:n_val]}

    val_traj = [t for t in trajectories if t["group"] in held]
    train_traj = [t for t in trajectories if t["group"] not in held]
    if not val_traj or not train_traj:
        raise RuntimeError(f"Podzial nie wyszedl: {len(groups)} grup. "
                           "Zbierz wiecej epizodow albo zmniejsz NUM_HELDOUT_GROUPS.")

    state_dim = trajectories[0]["states"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    state_mean, state_std = compute_state_normalization(train_traj)
    mark_turn_indices(trajectories)

    print(f"Urzadzenie: {DEVICE}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"  state_dim={state_dim} (po sin/cos yaw), act_dim={act_dim}, "
          f"action_scale={action_scale:.4f} m")
    print(f"  trening: {len(train_traj)} sekwencji, walidacja: {len(val_traj)} "
          f"({len(groups)} grup, odlozone: {sorted(held)})")

    # trywialna linia bazowa: zawsze prosto
    all_a = np.concatenate([t["actions"][t["valid"]] for t in train_traj])
    base = np.abs(np.degrees(np.arctan2(all_a[:, 0], all_a[:, 1])))
    print(f"  linia bazowa 'zawsze prosto': sredni blad kierunku {base.mean():.1f} deg, "
          f"trafien <=15 deg {100*(base <= 15).mean():.0f}%")

    model = WaypointTransformer(
        state_dim=state_dim, context_length=CONTEXT_LENGTH, hidden_size=HIDDEN_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, dropout=DROPOUT,
        n_dir_bins=N_DIR_BINS, mag_weight=MAG_WEIGHT, label_smooth=LABEL_SMOOTH,
    ).to(DEVICE)
    print(f"  parametrow: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min((it + 1) / WARMUP_ITERS, 1.0))

    rng = np.random.RandomState(SPLIT_SEED + 1000)
    args = (BATCH_SIZE, CONTEXT_LENGTH, state_dim, act_dim,
            state_mean, state_std, DEVICE)

    best_val, best_state, since_best = float("inf"), None, 0

    for it in range(1, NUM_TRAIN_ITERS + 1):
        model.train()
        s, a, m, v = get_batch(train_traj, *args, rng)
        loss, ce, mg = model.compute_loss(s, m, a, m * v)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
        opt.step()
        sched.step()

        if it % VAL_EVERY == 0 or it == 1:
            model.eval()
            vl, ang, med, hit, mge = [], [], [], [], []
            with torch.no_grad():
                for _ in range(VAL_BATCHES):
                    s, a, m, v = get_batch(val_traj, *args, rng)
                    lm = m * v
                    l, _, _ = model.compute_loss(s, m, a, lm)
                    vl.append(l.item())
                    x = angular_metrics(model, s, m, a, lm, N_DIR_BINS)
                    ang.append(x[0]); med.append(x[1]); hit.append(x[2]); mge.append(x[3])
            vloss = float(np.mean(vl))
            print(f"[{it:5d}/{NUM_TRAIN_ITERS}] train={loss.item():.4f} val={vloss:.4f} | "
                  f"blad kierunku: sr={np.mean(ang):5.1f} deg med={np.mean(med):5.1f} deg "
                  f"| <=15 deg: {100*np.mean(hit):4.0f}% | blad dlugosci={np.mean(mge)*action_scale:.2f} m")

            if vloss < best_val:
                best_val, since_best = vloss, 0
                best_state = {k: t.detach().cpu().clone()
                              for k, t in model.state_dict().items()}
            else:
                since_best += 1
                if since_best >= EARLY_STOP_PATIENCE:
                    print(f"Early stop: brak poprawy przez {EARLY_STOP_PATIENCE} walidacji.")
                    break
        elif it % LOG_EVERY == 0:
            print(f"[{it:5d}/{NUM_TRAIN_ITERS}] train={loss.item():.4f} "
                  f"(ce={ce.item():.4f} mag={mg.item():.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    report_by_turn_bucket(model, val_traj, args,
                          np.random.RandomState(SPLIT_SEED + 7), action_scale)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "held_out_groups": sorted(held),
        "best_val_loss": best_val,
        "config": {
            "state_dim": state_dim, "act_dim": act_dim,
            "context_length": CONTEXT_LENGTH, "hidden_size": HIDDEN_SIZE,
            "n_layer": N_LAYER, "n_head": N_HEAD,
            "n_dir_bins": N_DIR_BINS, "mag_weight": MAG_WEIGHT,
            "label_smooth": LABEL_SMOOTH,
            "action_scale": action_scale,
            "use_yaw_sincos": USE_YAW_SINCOS, "yaw_index": yaw_index,
            "state_clip": STATE_CLIP,
            "turn_oversample": TURN_OVERSAMPLE,
            "state_columns": state_columns,
            "dataset_file": DATASET_FILE,
        },
    }
    torch.save(ckpt, CHECKPOINT_FILE)
    print(f"\nZapisano checkpoint (najlepszy val={best_val:.4f}) do {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()