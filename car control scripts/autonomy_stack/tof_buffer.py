import time

class TofScanBuffer:
    def __init__(self, sector_count = 16, min_yaw = -87.5, max_yaw = 76.5, max_age = 5.0, max_dist = 3000.0):
        self.sector_count = sector_count
        self.min_yaw = min_yaw
        self.max_yaw = max_yaw
        self.max_age = max_age
        self.max_dist = max_dist

        self.distances = [0.0] * sector_count
        self.pitches = [0.0] * sector_count
        self.measurement_times = [0.0] * sector_count
        self.has_measurement = [False] * sector_count

    def update(self, pitch_deg, yaw_deg, distance_mm):
        # Normalize yaw to 0.0 - 1.0 range based on physical servo limits
        norm_yaw = (yaw_deg - self.min_yaw) / (self.max_yaw - self.min_yaw)
        norm_yaw = max(0.0, min(1.0, norm_yaw))
        
        # Drop into one of the 16 sectors
        sector = min(int(norm_yaw * self.sector_count), self.sector_count - 1)
        
        self.distances[sector] = distance_mm
        self.pitches[sector] = pitch_deg
        self.measurement_times[sector] = time.perf_counter()
        self.has_measurement[sector] = True

    def get_normalized_distances(self):
        now = time.perf_counter()
        return [
            max(0.0, min(1.0, self.distances[s] / self.max_dist))
            if self.has_measurement[s] and (now - self.measurement_times[s]) < self.max_age else 1.0
            for s in range(self.sector_count)
        ]

    def get_normalized_ages(self):
        now = time.perf_counter()
        return [
            max(0.0, min(1.0, (now - self.measurement_times[s]) / self.max_age))
            if self.has_measurement[s] else 1.0
            for s in range(self.sector_count)
        ]

    def get_measurement_pitches_deg(self):
        now = time.perf_counter()
        return [
            self.pitches[s]
            if self.has_measurement[s] and (now - self.measurement_times[s]) < self.max_age else 0.0
            for s in range(self.sector_count)
        ]
    