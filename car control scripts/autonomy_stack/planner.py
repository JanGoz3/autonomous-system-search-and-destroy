import collections
import math
import numpy as np

class TransformerPlanner():
    def __init__(self, session, context_length = 20, state_dim=70):
        self.session = session
        self.context_length = context_length
        self.state_dim = state_dim
        self.n_dir_bins = 36

        self.decision_interval = 0.5
        self.max_command_angle_deg = 70.0
        self.use_soft_direction = True
        self.soft_top_k = 5
        self.straight_deadzone_deg = 8.0
        self.min_cone_mass = 0.15

        inputs = self.session.get_inputs()
        self.states_name = inputs[0].name
        self.mask_name = inputs[1].name

        self.state_history = collections.deque(maxlen=self.context_length)
        self.last_decision_time = 0.0
        self.last_global_target_x = 0.0
        self.last_global_target_z = 0.0        

    def build_70d_state(self, car_yaw_rad, raw_telemetry, yolo_obs, tof_buffer):
        state = []
        state.append(math.degrees(car_yaw_rad))
        state.extend(raw_telemetry[4:11])

        for slot in range(2):
            base_idx = slot * 9
            state.extend(yolo_obs[base_idx : base_idx + 7])

        state.extend(tof_buffer.get_normalized_distances())
        state.extend(tof_buffer.get_normalized_ages())

        pitches = tof_buffer.get_measurement_pitches_deg()
        state.extend([p / 45.0 for p in pitches])

        return np.array(state, dtype = np.float32)

    def _pick_direction(self, probs):
        valid_indices = []
        cone_mass = 0.0

        for i in range(self.n_dir_bins):
            angle = (i + 0.5) * (360.0 / self.n_dir_bins) - 180.0
            if abs(angle) <= self.max_command_angle_deg:
                valid_indices.append(i)
                cone_mass += probs[i]

        if cone_mass < self.min_cone_mass or not valid_indices:
            return None, cone_mass

        if self.use_soft_direction:
            valid_indices.sort(key = lambda x: probs[x], reverse=True)
            top_k = valid_indices[:self.soft_top_k]

            sx, sy, w = 0.0, 0.0, 0.0
            for idx in top_k:
                p = probs[idx]
                a_rad = math.radians((idx + 0.5) * (360.0 / self.n_dir_bins) - 180.0)
                sx += p * math.sin(a_rad)
                sy += p * math.cos(a_rad)
                w += p

            final_angle = math.degrees(math.atan2(sx, sy)) if w >= 1e-6 else 0.0
        else:
            best_idx = max(valid_indices, key=lambda x: probs[x])
            final_angle = (best_idx + 0.5) * (360.0 / self.n_dir_bins) - 180.0

        if abs(final_angle) < self.straight_deadzone_deg:
            final_angle = 0.0

        return final_angle, cone_mass

    def get_destination(self, current_time, car_yaw_rad, raw_telemetry, yolo_obs, tof_buffer, car_global_x, car_global_z):
        current_state = self.build_70d_state(car_yaw_rad, raw_telemetry, yolo_obs, tof_buffer)        

        if current_time - self.last_decision_time < self.decision_interval:
            return self.last_global_target_x, self.last_global_target_z
        
        self.state_history.append(current_state)
        seq_len = len(self.state_history)
        pad_len = self.context_length - seq_len

        states_arr = np.zeros((1, self.context_length, self.state_dim), dtype=np.float32)
        mask_arr = np.zeros((1, self.context_length), dtype=np.float32)

        for i in range(seq_len):
            states_arr[0, pad_len + i, :] = self.state_history[i]
            mask_arr[0, pad_len + i] = 1.0

        outputs = self.session.run(["dir_probs", "mag_m"], {
            self.states_name: states_arr,
            self.mask_name: mask_arr
        })

        dir_probs = outputs[0][0]   # Shape (36,)
        mag_m = outputs[1][0][0]    # Shape (1,)

        angle_deg, cone_mass = self._pick_direction(dir_probs)
        
        if angle_deg is None:
            self.last_decision_time = current_time
            return self.last_global_target_x, self.last_global_target_z
        
        mag_m = float(np.clip(mag_m, 0.3, 1.5))
        angle_rad = math.radians(angle_deg)
        
        local_dx = mag_m * math.sin(angle_rad)
        local_dz = mag_m * math.cos(angle_rad)
        
        cos_y = math.cos(car_yaw_rad)
        sin_y = math.sin(car_yaw_rad)
        
        global_error_x = local_dx * cos_y + local_dz * sin_y
        global_error_z = -local_dx * sin_y + local_dz * cos_y

        self.last_global_target_x = car_global_x + global_error_x
        self.last_global_target_z = car_global_z + global_error_z
        self.last_decision_time = current_time
        
        return self.last_global_target_x, self.last_global_target_z
