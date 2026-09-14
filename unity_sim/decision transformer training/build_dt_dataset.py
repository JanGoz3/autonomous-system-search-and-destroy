import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\Admin\AppData\LocalLow\DefaultCompany\Search and destroy\DTDatasetBC"

INCLUDE_POSITION = False
INCLUDE_SCAN = True
INCLUDE_SCAN_PITCH = True
EXCLUDE_POLICY_OUTPUTS = True          # telem_0..3 = wyjscia polityki PPO
POLICY_OUTPUT_COLUMNS = ("telem_0", "telem_1", "telem_2", "telem_3")

YOLO_FIRST_INDEX = 11
YOLO_FEATURES_PER_SLOT = 9
YOLO_MAX_SLOTS = 2
YOLO_DROP_OFFSETS = (7, 8)
SCAN_PITCH_SCALE_DEG = 45.0

ADD_DIRECTION_CHANNEL = False
REVERSE_MARKER = "_rev_"

MAX_LABEL_ANGLE_DEG = 70.0
MAX_PROGRESS_JUMP_M = 1.0
MAX_DEVIATION_M = 2.5

STILL_WINDOW = 5
STILL_DIST_M = 0.02
KEEP_EVERY_STILL = 10
MIN_VALID_FRACTION = 0.30

DECIMATE = 5

FILE_FILTER = None
KEEP_ALL_PHASES = True
MIN_DECIMATED_LENGTH = 25
MIN_EPISODE_LENGTH = 200

SUBSAMPLE_KEYS = ("states", "actions_m", "valid")

OUTPUT_FILE = (f"bc_dataset{'_' + FILE_FILTER.strip('_') if FILE_FILTER else ''}"
               f"_pos{int(INCLUDE_POSITION)}"
               f"_scan{int(INCLUDE_SCAN)}{'p' if INCLUDE_SCAN_PITCH else ''}"
               f"{'_noyolo' if YOLO_MAX_SLOTS == 0 else f'_yolo{YOLO_MAX_SLOTS}'}"
               f"{'_dir' if ADD_DIRECTION_CHANNEL else ''}.pkl")


def indexed_columns(df, prefix):
    cols = sorted((c for c in df.columns if re.fullmatch(rf"{prefix}[0-9]+", c)),
                  key=lambda c: int(c[len(prefix):]))
    expected = [f"{prefix}{i}" for i in range(len(cols))]
    if cols != expected:
        raise ValueError(f"Nieciagle indeksy kolumn {prefix} w CSV: {cols}")
    return cols


def get_state_columns(df):
    telem = indexed_columns(df, "telem_")
    if not telem:
        raise ValueError("Brakuje kolumn telemetrii w CSV")

    if EXCLUDE_POLICY_OUTPUTS:
        telem = [c for c in telem if c not in POLICY_OUTPUT_COLUMNS]
    def keep_telem(name):
        i = int(name[len("telem_"):])
        if i < YOLO_FIRST_INDEX:
            return True
        slot, off = divmod(i - YOLO_FIRST_INDEX, YOLO_FEATURES_PER_SLOT)
        return slot < YOLO_MAX_SLOTS and off not in YOLO_DROP_OFFSETS

    telem = [c for c in telem if keep_telem(c)]

    cols = (["direction"] if ADD_DIRECTION_CHANNEL else []) \
         + (["posX", "posZ"] if INCLUDE_POSITION else []) + ["yaw"] + telem
    pitch_cols = []

    if INCLUDE_SCAN:
        dist = indexed_columns(df, "scan_dist_")
        ages = indexed_columns(df, "scan_age_")
        if not dist and not ages:
            warnings.warn("INCLUDE_SCAN=True, ale CSV nie ma scan_dist_*/scan_age_*.")
        elif len(dist) != len(ages):
            raise ValueError("Liczba kolumn scan_dist_* != scan_age_*")
        else:
            cols += dist + ages
            if INCLUDE_SCAN_PITCH:
                pitches = indexed_columns(df, "scan_pitch_")
                if not pitches:
                    warnings.warn("Brak scan_pitch_* - stan bez pitcha skanu.")
                elif len(pitches) != len(dist):
                    raise ValueError("Liczba scan_pitch_* != scan_dist_*")
                else:
                    cols += pitches
                    pitch_cols = pitches

    return cols, pitch_cols


def build_state_vector(df, cols, pitch_cols):
    states = df[cols].to_numpy(dtype=np.float32)
    if pitch_cols:
        idx = [cols.index(c) for c in pitch_cols]
        states[:, idx] /= SCAN_PITCH_SCALE_DEG
    return states


def moving_mask(pos, window=STILL_WINDOW, thr=STILL_DIST_M):
    fut = np.minimum(np.arange(len(pos)) + window, len(pos) - 1)
    return np.hypot(*(pos[fut] - pos).T) >= thr


def thin_still_runs(moving, keep_every=KEEP_EVERY_STILL):
    valid = moving.copy()
    n, i = len(moving), 0
    while i < n:
        if moving[i]:
            i += 1
            continue
        j = i
        while j < n and not moving[j]:
            j += 1
        valid[i:j:keep_every] = True
        i = j
    return valid


def label_masks(df, actions_m):
    n = len(df)
    stats = {}

    ok = (df["expert_valid"].to_numpy().astype(bool)
          if "expert_valid" in df.columns else np.ones(n, dtype=bool))
    stats["expert_valid=0"] = int((~ok).sum())

    mag = np.hypot(actions_m[:, 0], actions_m[:, 1])
    m_mag = mag > 0.05
    stats["dlugosc < 0.05 m"] = int((~m_mag).sum())

    ang = np.degrees(np.arctan2(actions_m[:, 0], actions_m[:, 1]))
    m_ang = np.abs(ang) <= MAX_LABEL_ANGLE_DEG
    stats[f"|kat| > {MAX_LABEL_ANGLE_DEG:.0f} deg"] = int((~m_ang).sum())

    if "progress_along_route" in df.columns:
        prog = df["progress_along_route"].to_numpy(dtype=np.float64)
        jump = np.abs(np.r_[0.0, np.diff(prog)])
        m_jump = jump <= MAX_PROGRESS_JUMP_M
        stats[f"skok postepu > {MAX_PROGRESS_JUMP_M} m"] = int((~m_jump).sum())
    else:
        m_jump = np.ones(n, dtype=bool)
        stats["skok postepu"] = "brak kolumny progress_along_route"

    if "deviation_from_route" in df.columns:
        dev = df["deviation_from_route"].to_numpy(dtype=np.float64)
        m_dev = dev <= MAX_DEVIATION_M
        stats[f"odchylenie > {MAX_DEVIATION_M} m"] = int((~m_dev).sum())
    else:
        m_dev = np.ones(n, dtype=bool)
        stats["odchylenie"] = "brak kolumny deviation_from_route"

    pos = df[["posX", "posZ"]].to_numpy(dtype=np.float64)
    m_move = thin_still_runs(moving_mask(pos))
    stats["bezruch (po przerzedzeniu)"] = int((~m_move).sum())

    valid = ok & m_mag & m_ang & m_jump & m_dev & m_move
    return valid, stats


def process_episode(path: Path):
    df = pd.read_csv(path)
    if ADD_DIRECTION_CHANNEL:
        df["direction"] = -1.0 if REVERSE_MARKER in path.name else 1.0
    if len(df) < MIN_EPISODE_LENGTH:
        print(f"  [pominieto] {path.name}: za krotki ({len(df)} < {MIN_EPISODE_LENGTH})")
        return None

    if not {"expert_x", "expert_z"} <= set(df.columns):
        raise ValueError(f"{path.name}: brak kolumn expert_x/expert_z. "
                         "Droga A wymaga etykiet eksperckich.")

    actions_m = df[["expert_x", "expert_z"]].to_numpy(dtype=np.float64)
    valid, stats = label_masks(df, actions_m)

    frac = valid.mean()
    if frac < MIN_VALID_FRACTION:
        print(f"  [pominieto] {path.name}: tylko {100*frac:.0f}% waznych probek")
        for k, v in stats.items():
            print(f"      odrzucone przez {k}: {v}")
        return None

    cols, pitch_cols = get_state_columns(df)
    return {
        "states": build_state_vector(df, cols, pitch_cols),
        "state_columns": cols,
        "actions_m": actions_m.astype(np.float32),
        "valid": valid,
        "episode_length": len(df),
        "source_file": path.name,
        "group": path.name.split("_episode_")[0] if "_episode_" in path.name else path.name,
        "reject_stats": stats,
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
    for t in trajectories:
        for ph in phases:
            d = decimate(t, DECIMATE, ph)
            if d["episode_length"] >= MIN_DECIMATED_LENGTH:
                out.append(d)
    return out


def report_scan_health(trajectories, cols):
    dist_idx = [i for i, c in enumerate(cols) if c.startswith("scan_dist_")]
    age_idx = [i for i, c in enumerate(cols) if c.startswith("scan_age_")]
    if not dist_idx:
        return
    S = np.concatenate([t["states"] for t in trajectories])
    ages = S[:, age_idx]
    fresh = (ages < 0.999).sum(axis=1)
    print(f"\nProfil ToF ({len(dist_idx)} sektorow):")
    print(f"  sektorow z POMIAREM na krok: srednia={fresh.mean():.1f} mediana={np.median(fresh):.0f} "
          f"min={fresh.min()} max={fresh.max()}")
    print(f"  udzial sektorow przeterminowanych (age>=0.999): {100*(ages >= 0.999).mean():.0f}%")
    if fresh.mean() < 2:
        print("  UWAGA: bufor nie akumuluje - sprawdz, czy wiezyczka omiata zakres.")


def report_yolo_health(trajectories, cols):
    if YOLO_MAX_SLOTS == 0:
        return
    S = np.concatenate([t["states"] for t in trajectories])
    print(f"\nBlok YOLO ({YOLO_MAX_SLOTS} slotow):")
    for slot in range(YOLO_MAX_SLOTS):
        base = YOLO_FIRST_INDEX + slot * YOLO_FEATURES_PER_SLOT
        conf_name = f"telem_{base + 4}"
        if conf_name not in cols:
            continue
        conf = S[:, cols.index(conf_name)]
        det = conf > 0
        line = f"  slot {slot}: detekcja w {100 * det.mean():5.1f}% krokow"
        door_name = f"telem_{base + 6}"          # flaga CLASS_DOOR
        if door_name in cols:
            door = S[:, cols.index(door_name)]
            line += f", drzwi w {100 * (door > 0).mean():5.1f}%"
        print(line)
    if YOLO_MAX_SLOTS >= 3:
        print("  UWAGA: slot 2 odpalal sie w 0.3% probek i dawal |z| = 44.7 - "
              "rozwaz YOLO_MAX_SLOTS = 2")


def report_degenerate_columns(trajectories, cols):
    S = np.concatenate([t["states"] for t in trajectories]).astype(np.float64)
    sd = S.std(axis=0)
    mu = S.mean(axis=0)
    z = np.abs((S - mu) / (sd + 1e-6)).max(axis=0)

    dead = [cols[i] for i in np.where(sd < 1e-6)[0]]
    spiky = [(cols[i], z[i]) for i in np.argsort(-z)[:5] if z[i] > 15]

    print(f"\nKondycja wektora stanu ({len(cols)} wymiarow):")
    print(f"  max |z| po normalizacji: {z.max():.1f}")
    if dead:
        print(f"  KOLUMNY STALE ({len(dead)}): {', '.join(dead)}")
        print("    -> martwe wymiary, rozwaz wyciecie")
    if spiky:
        print("  KOLUMNY Z EKSTREMAMI (|z| > 15):")
        for c, v in spiky:
            print(f"    {c}: max|z|={v:.1f}")
    if not dead and not spiky:
        print("  brak kolumn stalych i brak ekstremow - OK")


def main():
    files = sorted(Path(DATA_DIR).glob("*episode_*.csv"))
    if FILE_FILTER:
        before = len(files)
        files = [f for f in files if FILE_FILTER in f.name]
        print(f"FILE_FILTER = '{FILE_FILTER}': {before} -> {len(files)} plikow")
    print(f"Znaleziono {len(files)} plikow CSV w {DATA_DIR}")
    if not files:
        print("Sprawdz DATA_DIR.")
        return

    trajectories = []
    agg_stats = {}
    for p in files:
        r = process_episode(p)
        if r is None:
            continue
        if trajectories and r["state_columns"] != trajectories[0]["state_columns"]:
            raise ValueError(f"{p.name}: inny schemat stanu niz w poprzednich CSV.")
        for k, v in r["reject_stats"].items():
            if isinstance(v, int):
                agg_stats[k] = agg_stats.get(k, 0) + v
        trajectories.append(r)
        print(f"  {p.name}: {r['episode_length']} krokow, wazne={100*r['valid'].mean():.0f}%")

    if not trajectories:
        print("Zaden epizod nie przeszedl - nic nie zapisano.")
        return

    total_raw = sum(t["episode_length"] for t in trajectories)
    print(f"\n--- Odrzucone probki (na {total_raw} krokow surowych) ---")
    for k, v in sorted(agg_stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {v:6d}  ({100*v/total_raw:.1f}%)")

    if DECIMATE > 1:
        before = len(trajectories)
        trajectories = expand_phases(trajectories)
        n_groups = len({t["group"] for t in trajectories})
        print(f"\nDecymacja {DECIMATE}x: {before} epizodow -> {len(trajectories)} sekwencji "
              f"(~{trajectories[0]['episode_length']} krokow, {n_groups} grup)")

    all_valid = np.concatenate([t["actions_m"][t["valid"]] for t in trajectories])
    if len(all_valid) == 0:
        raise RuntimeError("Zero waznych probek po filtrach.")

    mag = np.hypot(all_valid[:, 0], all_valid[:, 1])
    action_scale = float(mag.mean())
    if action_scale < 1e-6:
        raise RuntimeError("Srednia dlugosc akcji ~0 - dane sa zle.")

    for t in trajectories:
        t["actions"] = (t["actions_m"] / action_scale).astype(np.float32)

    cols = trajectories[0]["state_columns"]
    n_valid = sum(int(t["valid"].sum()) for t in trajectories)
    total = sum(t["episode_length"] for t in trajectories)
    ang = np.degrees(np.arctan2(all_valid[:, 0], all_valid[:, 1]))

    print(f"\n--- Podsumowanie ---")
    print(f"state_dim={len(cols)}  yaw_index={cols.index('yaw')}")
    if ADD_DIRECTION_CHANNEL:
        n_rev = sum(1 for t in trajectories if REVERSE_MARKER in t["source_file"])
        print(f"  kanal kierunku: kolumna 0, {len(trajectories) - n_rev} sekwencji +1 "
              f"/ {n_rev} sekwencji -1")
    print(f"  wariant: pos={INCLUDE_POSITION} scan={INCLUDE_SCAN} "
          f"scan_pitch={INCLUDE_SCAN_PITCH} yolo_slotow={YOLO_MAX_SLOTS}")
    print(f"Sekwencji: {len(trajectories)}, krokow: {total}, waznych: {n_valid} "
          f"({100*n_valid/total:.0f}%)")
    print(f"ACTION_SCALE = {action_scale:.4f} m")
    print(f"Dlugosc etykiety [m]: mediana={np.median(mag):.2f} p90={np.quantile(mag,0.9):.2f} "
          f"max={mag.max():.2f}")
    print(f"Kat etykiety [deg]: std={ang.std():.1f} p5={np.percentile(ang,5):.0f} "
          f"p95={np.percentile(ang,95):.0f} |kat|>45: {100*(np.abs(ang)>45).mean():.1f}%")

    report_scan_health(trajectories, cols)
    report_yolo_health(trajectories, cols)
    report_degenerate_columns(trajectories, cols)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({
            "trajectories": trajectories,
            "state_columns": cols,
            "action_scale": action_scale,
            "include_position": INCLUDE_POSITION,
            "include_scan": INCLUDE_SCAN,
            "include_scan_pitch": INCLUDE_SCAN_PITCH,
            "exclude_policy_outputs": EXCLUDE_POLICY_OUTPUTS,
            "yolo_max_slots": YOLO_MAX_SLOTS,
            "yolo_first_index": YOLO_FIRST_INDEX,
            "yolo_features_per_slot": YOLO_FEATURES_PER_SLOT,
            "yolo_drop_offsets": list(YOLO_DROP_OFFSETS),
            "scan_pitch_scale_deg": SCAN_PITCH_SCALE_DEG,
            "add_direction_channel": ADD_DIRECTION_CHANNEL,
            "max_label_angle_deg": MAX_LABEL_ANGLE_DEG,
            "decimate": DECIMATE,
        }, f)
    print(f"\nZapisano do {OUTPUT_FILE}")


if __name__ == "__main__":
    main()