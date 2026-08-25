import pickle
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = r"C:\Users\Admin\AppData\LocalLow\DefaultCompany\Search and destroy\DTDataset"
OUTPUT_FILE = "dt_dataset.pkl"


HORIZON_STEPS = 15          # ile kroków w przyszłość patrzymy
MAX_ARENA_SIZE = 20.0        # ta sama stała co w CarAgent.cs
GRID_CELL_SIZE = 1.0         # rozmiar komórki siatki (metry) do liczenia pokrycia terenu

# UWAGA: kara za kolizje oparta o ToF (telem_10) zostala WYLACZONA.
# ToF jest fizycznie podpiety pod obracana wiezyczke (Turret), wiec nie mierzy
# odleglosci "do przodu" wzgledem kierunku jazdy, tylko tam gdzie akurat
# celuje wiezyczka - dlatego nie nadaje sie (na razie) do kary za kolizje
# nawigacyjne. Do przemyslenia pozniej w ramach reward shapingu.

COVERAGE_REWARD = 1.0        # nagroda za odwiedzenie nowej komórki siatki
STEP_PENALTY = -0.01         # mała kara za każdy krok (zachęca do ruchu/efektywności)

MIN_EPISODE_LENGTH = HORIZON_STEPS + 5  # minimalna długość epizodu (liczba kroków) żeby go wziąć pod uwagę przy budowie datasetu

STILLNESS_THRESHOLD_M = 0.001  # ruch miedzy kolejnymi krokami ponizej tej wartosci (metry) uznajemy za "bezruch"
MAX_STILL_FRACTION = 0.35      # jesli najdluzszy CIAGLY odcinek bezruchu przekracza ten % dlugosci
                                # epizodu, epizod jest odrzucany (auto najprawdopodobniej utkneło)

NUM_TELEMETRY_COLS = 17


def relabel_actions(df: pd.DataFrame, horizon_steps: int = HORIZON_STEPS) -> np.ndarray:

    n = len(df)
    actions = np.zeros((n, 2), dtype=np.float32)

    pos = df[["posX", "posZ"]].to_numpy()
    yaw = df["yaw"].to_numpy()

    for i in range(n):
        future_idx = min(i + horizon_steps, n - 1)

        dx = pos[future_idx, 0] - pos[i, 0]
        dz = pos[future_idx, 1] - pos[i, 1]

        # obrót wektora (dx,dz) o -yaw, żeby wyrazić go w lokalnym układzie auta
        yaw_rad = np.radians(yaw[i])
        cos_y, sin_y = np.cos(-yaw_rad), np.sin(-yaw_rad)
        local_x = dx * cos_y - dz * sin_y
        local_z = dx * sin_y + dz * cos_y

        actions[i, 0] = local_x / MAX_ARENA_SIZE
        actions[i, 1] = local_z / MAX_ARENA_SIZE

    return actions

def compute_rewards(df: pd.DataFrame, grid_cell_size: float = GRID_CELL_SIZE) -> np.ndarray:

    n = len(df)
    rewards = np.zeros(n, dtype=np.float32)

    pos = df[["posX", "posZ"]].to_numpy()

    visited_cells = set()

    for i in range(n):
        cell = (
            int(np.floor(pos[i, 0] / grid_cell_size)),
            int(np.floor(pos[i, 1] / grid_cell_size)),
        )

        r = STEP_PENALTY

        if cell not in visited_cells:
            visited_cells.add(cell)
            r += COVERAGE_REWARD

        rewards[i] = r

    return rewards


def compute_return_to_go(rewards: np.ndarray) -> np.ndarray:
    rtg = np.zeros_like(rewards)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running += rewards[i]
        rtg[i] = running
    return rtg

def build_state_vector(df: pd.DataFrame) -> np.ndarray:
    telem_cols = [f"telem_{i}" for i in range(NUM_TELEMETRY_COLS)]
    missing = [c for c in telem_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn telemetrii w CSV: {missing}")

    cols = ["posX", "posZ", "yaw"] + telem_cols
    return df[cols].to_numpy(dtype=np.float32)


def longest_still_run(df: pd.DataFrame, threshold_m: float = STILLNESS_THRESHOLD_M) -> int:
    pos = df[["posX", "posZ"]].to_numpy()
    if len(pos) < 2:
        return 0
    diffs = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    still = diffs < threshold_m

    max_run = 0
    cur_run = 0
    for s in still:
        if s:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run

def process_episode(csv_path: Path, horizon_steps: int = HORIZON_STEPS):
    df = pd.read_csv(csv_path)

    if len(df) < MIN_EPISODE_LENGTH:
        print(f"  [pominięto] {csv_path.name}: za krótki ({len(df)} krokow)")
        return None

    still_run = longest_still_run(df)
    still_fraction = still_run / len(df)

    if still_fraction > MAX_STILL_FRACTION:
        print(
            f"  [pominięto] {csv_path.name}: najdluzszy odcinek bezruchu = {still_run} krokow "
            f"({100*still_fraction:.0f}% epizodu) - auto najprawdopodobniej utkneło, epizod odrzucony"
        )
        return None

    states = build_state_vector(df)
    actions = relabel_actions(df, horizon_steps)
    rewards = compute_rewards(df)
    rtg = compute_return_to_go(rewards)

    return {
        "states": states,              # (T, 20)
        "actions": actions,            # (T, 2)
        "rewards": rewards,            # (T,)
        "returns_to_go": rtg,          # (T,)
        "episode_length": len(df),
        "source_file": csv_path.name,
        "longest_still_run": still_run,
    }

def main():
    data_dir = Path(DATA_DIR)
    csv_files = sorted(data_dir.glob("episode_*.csv"))

    print(f"Znaleziono {len(csv_files)} plikow CSV w {data_dir}")
    if len(csv_files) == 0:
        print("Brak plikow do przetworzenia - sprawdz sciezke DATA_DIR na gorze skryptu.")
        return

    trajectories = []
    for csv_path in csv_files:
        result = process_episode(csv_path, horizon_steps=HORIZON_STEPS)
        if result is not None:
            trajectories.append(result)
            print(
                f"  {csv_path.name}: {result['episode_length']} krokow, "
                f"suma nagrod={result['rewards'].sum():.2f}, "
                f"return_to_go[0]={result['returns_to_go'][0]:.2f}"
            )

    print(f"\nPoprawnie przetworzono {len(trajectories)} trajektorii")
    total_steps = sum(t["episode_length"] for t in trajectories)
    print(f"Laczna liczba krokow we wszystkich trajektoriach: {total_steps}")

    if len(trajectories) == 0:
        print("Zaden epizod nie przeszedl przetwarzania - nic nie zapisano.")
        return

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Zapisano dataset do {OUTPUT_FILE}")


if __name__ == "__main__":
    main()