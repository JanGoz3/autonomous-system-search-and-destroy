import pickle
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = r"C:\Users\Admin\AppData\LocalLow\DefaultCompany\Search and destroy\DTDataset"
OUTPUT_FILE = "dt_dataset.pkl"

USE_EXPERT_LABEL = True

USE_DISTANCE_RELABEL = True
WAYPOINT_DIST = 1.5          # metry - jak daleko ma byc waypoint
MAX_LOOKAHEAD_STEPS = 60     # 6 s przy 10 Hz - twardy limit szukania

HORIZON_STEPS = 15           # uzywane tylko gdy USE_DISTANCE_RELABEL = False

DECIMATE = 15
KEEP_ALL_PHASES = True

SUBSAMPLE_KEYS = ("states", "actions_m", "valid", "reached", "moving",
                  "rewards", "returns_to_go")

GRID_CELL_SIZE = 1.0

COVERAGE_REWARD = 1.0
STEP_PENALTY = -0.01
COLLISION_PENALTY = -2.0

MIN_EPISODE_LENGTH = MAX_LOOKAHEAD_STEPS + 10

STILL_WINDOW = 5
STILL_DIST_M = 0.02          # metry - ponizej tego w oknie = auto stoi

KEEP_EVERY_STILL = 10        # z odcinka bezruchu zostaw co N-ta probke
MIN_VALID_FRACTION = 0.10

NUM_TELEMETRY_COLS = 17


def world_to_local(d: np.ndarray, yaw_deg: np.ndarray) -> np.ndarray:
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    return np.stack([d[:, 0] * c - d[:, 1] * s,
                     d[:, 0] * s + d[:, 1] * c], axis=1)


def local_to_world(d: np.ndarray, yaw_deg: np.ndarray) -> np.ndarray:
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    return np.stack([d[:, 0] * c + d[:, 1] * s,
                     -d[:, 0] * s + d[:, 1] * c], axis=1)

def relabel_by_distance(pos, yaw, target=WAYPOINT_DIST, max_steps=MAX_LOOKAHEAD_STEPS):
    n = len(pos)
    j_idx = np.empty(n, dtype=np.int64)
    reached = np.zeros(n, dtype=bool)

    for i in range(n):
        hi = min(i + max_steps, n - 1)
        seg = pos[i + 1:hi + 1] - pos[i]
        if len(seg) == 0:
            j_idx[i] = i
            continue
        dist = np.hypot(seg[:, 0], seg[:, 1])
        hit = np.flatnonzero(dist >= target)
        if len(hit):
            j_idx[i] = i + 1 + hit[0]
            reached[i] = True
        else:
            j_idx[i] = hi

    return world_to_local(pos[j_idx] - pos, yaw), reached


def relabel_by_steps(pos, yaw, horizon=HORIZON_STEPS):
    n = len(pos)
    fut = np.minimum(np.arange(n) + horizon, n - 1)
    return world_to_local(pos[fut] - pos, yaw), np.ones(n, dtype=bool)


def moving_mask(pos, window=STILL_WINDOW, thr=STILL_DIST_M):
    n = len(pos)
    fut = np.minimum(np.arange(n) + window, n - 1)
    return np.hypot(*(pos[fut] - pos).T) >= thr


def thin_still_runs(moving, keep_every=KEEP_EVERY_STILL):
    valid = moving.copy()
    n = len(moving)
    i = 0
    while i < n:
        if moving[i]:
            i += 1
            continue
        j = i
        while j < n and not moving[j]:
            j += 1
        valid[i:j:keep_every] = True     # co keep_every-ta z odcinka [i, j)
        i = j
    return valid

def compute_rewards(df, grid_cell_size=GRID_CELL_SIZE):
    n = len(df)
    rewards = np.zeros(n, dtype=np.float32)
    pos = df[["posX", "posZ"]].to_numpy()
    collisions = (df["collision"].to_numpy() if "collision" in df.columns
                  else np.zeros(n, dtype=bool))

    visited = set()
    for i in range(n):
        cell = (int(np.floor(pos[i, 0] / grid_cell_size)),
                int(np.floor(pos[i, 1] / grid_cell_size)))
        r = STEP_PENALTY
        if cell not in visited:
            visited.add(cell)
            r += COVERAGE_REWARD
        if collisions[i]:
            r += COLLISION_PENALTY
        rewards[i] = r
    return rewards


def compute_return_to_go(rewards):
    return np.cumsum(rewards[::-1])[::-1].astype(np.float32).copy()


def build_state_vector(df):
    telem_cols = [f"telem_{i}" for i in range(NUM_TELEMETRY_COLS)]
    missing = [c for c in telem_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn telemetrii w CSV: {missing}")
    return df[["posX", "posZ", "yaw"] + telem_cols].to_numpy(dtype=np.float32)


def process_episode(csv_path: Path):
    df = pd.read_csv(csv_path)
    if len(df) < MIN_EPISODE_LENGTH:
        print(f"  [pominieto] {csv_path.name}: za krotki ({len(df)})")
        return None

    pos = df[["posX", "posZ"]].to_numpy(dtype=np.float64)
    yaw = df["yaw"].to_numpy(dtype=np.float64)

    has_expert = USE_EXPERT_LABEL and {"expert_x", "expert_z"} <= set(df.columns)
    if has_expert:
        actions_m = df[["expert_x", "expert_z"]].to_numpy(dtype=np.float64)
        reached = (df["expert_valid"].to_numpy().astype(bool)
                   if "expert_valid" in df.columns
                   else np.ones(len(df), dtype=bool))

        reached &= np.hypot(actions_m[:, 0], actions_m[:, 1]) > 0.05
    elif USE_DISTANCE_RELABEL:
        actions_m, reached = relabel_by_distance(pos, yaw)
    else:
        actions_m, reached = relabel_by_steps(pos, yaw)

    moving = moving_mask(pos)

    valid = (thin_still_runs(moving) if has_expert else moving) & reached
    valid_frac = valid.mean()

    if valid_frac < MIN_VALID_FRACTION:
        print(f"  [pominieto] {csv_path.name}: tylko {100 * valid_frac:.0f}% waznych "
              f"probek (auto glownie stalo/napieralo na sciane)")
        return None

    rewards = compute_rewards(df)
    return {
        "states": build_state_vector(df),          # (T, 20)
        "actions_m": actions_m.astype(np.float32),  # (T, 2) W METRACH
        "valid": valid,                             # (T,) bool - maska do lossu
        "reached": reached,
        "moving": moving,
        "rewards": rewards,
        "returns_to_go": compute_return_to_go(rewards),
        "episode_length": len(df),
        "source_file": csv_path.name,
        "group": csv_path.name,
        "n_collisions": int(df["collision"].sum()) if "collision" in df.columns else 0,
        "expert_label": bool(has_expert),
    }


def decimate(traj, factor, phase):
    idx = np.arange(phase, traj["episode_length"], factor)
    out = dict(traj)
    for k in SUBSAMPLE_KEYS:
        out[k] = traj[k][idx]
    out["episode_length"] = len(idx)
    out["source_file"] = f"{traj['source_file']}#p{phase:02d}"
    return out


def expand_phases(trajectories):
    if DECIMATE <= 1:
        return trajectories
    phases = range(DECIMATE) if KEEP_ALL_PHASES else [0]
    out = []
    for traj in trajectories:
        for ph in phases:
            d = decimate(traj, DECIMATE, ph)
            if d["episode_length"] >= 20:
                out.append(d)
    return out


def main():
    csv_files = sorted(Path(DATA_DIR).glob("episode_*.csv"))
    print(f"Znaleziono {len(csv_files)} plikow CSV")
    if not csv_files:
        print("Sprawdz DATA_DIR.")
        return

    trajectories = []
    for p in csv_files:
        r = process_episode(p)
        if r is None:
            continue
        trajectories.append(r)
        print(f"  {p.name}: {r['episode_length']} krokow, "
              f"wazne={100 * r['valid'].mean():.0f}%, "
              f"suma nagrod={r['rewards'].sum():.1f}, "
              f"kolizje={r['n_collisions']}")

    if not trajectories:
        print("Zaden epizod nie przeszedl - nic nie zapisano.")
        return

    if DECIMATE > 1:
        before = len(trajectories)
        trajectories = expand_phases(trajectories)
        print(f"\nDecymacja {DECIMATE}x ({'wszystkie fazy' if KEEP_ALL_PHASES else 'faza 0'}): "
              f"{before} epizodow -> {len(trajectories)} sekwencji po "
              f"~{trajectories[0]['episode_length']} krokow "
              f"({len(set(t['group'] for t in trajectories))} grup)")

    all_valid = np.concatenate([t["actions_m"][t["valid"]] for t in trajectories])

    use_expert = all(t.get("expert_label") for t in trajectories)
    action_scale = float(np.hypot(all_valid[:, 0], all_valid[:, 1]).mean()
                         if use_expert else all_valid.std())
    if action_scale < 1e-6:
        raise RuntimeError("std akcji ~0 - cos jest powaznie nie tak z danymi.")

    for t in trajectories:
        t["actions"] = (t["actions_m"] / action_scale).astype(np.float32)

    total = sum(t["episode_length"] for t in trajectories)
    n_valid = sum(int(t["valid"].sum()) for t in trajectories)
    mag = np.hypot(all_valid[:, 0], all_valid[:, 1])

    print(f"\n--- Podsumowanie ---")
    print(f"Zrodlo etykiet: {'waypoint eksperta (expert_x/expert_z)' if use_expert else 'relabeling z ruchu auta'}")
    print(f"Epizodow: {len(trajectories)}, krokow: {total}")
    print(f"Waznych probek: {n_valid} ({100 * n_valid / total:.0f}%)")
    print(f"ACTION_SCALE (std, metry): {action_scale:.4f}")
    print(f"Dlugosc akcji [m]: mediana={np.median(mag):.2f}, "
          f"p90={np.quantile(mag, 0.9):.2f}, max={mag.max():.2f}")
    print(f"Udzial akcji 'do tylu' (local_z < 0): "
          f"{100 * (all_valid[:, 1] < 0).mean():.0f}%")

    rtg = np.concatenate([t["returns_to_go"] for t in trajectories])
    print(f"\nReturn-to-go obserwowany w danych (do wpisania w DTInference):")
    print(f"  returnToGoMin = {np.percentile(rtg, 1):.0f}   "
          f"returnToGoMax = {np.percentile(rtg, 99):.0f}")
    print(f"  mediana={np.median(rtg):.1f}, zakres pelny "
          f"[{rtg.min():.0f}, {rtg.max():.0f}]")
    print(f"  initialTargetReturn ustaw blisko gornego konca, np. "
          f"{np.percentile(rtg, 90):.0f} - wyzej to ekstrapolacja poza dane.")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"trajectories": trajectories,
                     "action_scale": action_scale,
                     "waypoint_dist": WAYPOINT_DIST if USE_DISTANCE_RELABEL else None},
                    f)
    print(f"\nZapisano do {OUTPUT_FILE}")
    print("UWAGA: format pliku sie zmienil (dict zamiast listy). "
          "train_dt.py / evaluate_dt.py / test_ambiguity musza czytac ['trajectories'].")


if __name__ == "__main__":
    main()