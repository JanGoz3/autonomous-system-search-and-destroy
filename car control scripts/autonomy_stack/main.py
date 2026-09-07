import time
import threading
from collections import deque
import cv2
import numpy as np
import onnxruntime as ort
import os

from hardware import CarHardware
from perception import YoloProcessor, gstreamer_pipeline
from agents import DriverNetAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DRIVER_MODEL_PATH = os.path.join(BASE_DIR, "DriverNet.onnx")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo_ours_v4.onnx")


STACKED_VECTORS = 3
STATE_SPACE = 40
MAX_ARENA_SIZE = 20.0  
CONTROL_HZ = 20.0      
MAX_EXEC_TIME = 2.0
ACCEL_FACTOR = 0.15

current_target_x = 0.0
current_target_z = 0.0  
last_command_time = 0.0  
is_running = True

def terminal_input_thread():
    global current_target_x, current_target_z, is_running, last_command_time
    
    print("\n" + "="*50)
    print("TERMINAL TARGET CONTROL ACTIVE")
    print("Format: X, Z (e.g. '0.5, 2.0')")
    print("Type 'q' and press Enter to stop the car.")
    print("="*50 + "\n")
    
    while is_running:
        try:
            user_input = input()
            if user_input.strip().lower() == 'q':
                is_running = False
                break
                
            parts = user_input.split(',')
            if len(parts) == 2:
                current_target_x = float(parts[0].strip())
                current_target_z = float(parts[1].strip())
                last_command_time = time.perf_counter()
                print(f">>> [ACCEPTED] Car routing to X:{current_target_x:.2f}m, Z:{current_target_z:.2f}m")
            else:
                print(">>> [ERROR] Invalid format. Type exactly like: 0.5, 2.0")
        except ValueError:
            print(">>> [ERROR] Please enter valid numbers.")
        except EOFError:
            break

def main():
    global is_running, current_target_x, current_target_z, last_command_time
    
    print("Initializing Models and Hardware...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    
    yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=providers)
    yolo_processor = YoloProcessor(yolo_session)

    driver_session = ort.InferenceSession(DRIVER_MODEL_PATH, providers=providers)
    driver_agent = DriverNetAgent(driver_session)

    hardware = CarHardware("/dev/ttyACM0", 115200)
    
    print("Initializing CSI Camera via GStreamer...")
    pipeline = gstreamer_pipeline(flip_method=0)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open CSI physical camera.")

    frame_buffer = deque([np.zeros(STATE_SPACE, dtype=np.float32) for _ in range(STACKED_VECTORS)], maxlen=STACKED_VECTORS)
    loop_interval = 1.0 / CONTROL_HZ
    
    input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
    input_thread.start()

    loop_last_time = time.perf_counter()
    last_command_time = time.perf_counter()
    estimated_speed = 0.0

    try:
        while is_running:
            current_time = time.perf_counter()
            dt = current_time - loop_last_time
            loop_last_time = current_time

            ret, frame = cap.read()
            if not ret: continue

            yolo_obs = yolo_processor.process_frame(frame)
            raw_telemetry = hardware.get_telemetry()
            
            max_speed_mps = 2.0 
            target_speed = raw_telemetry[0] * max_speed_mps  
            
            estimated_speed = (estimated_speed * (1.0 - ACCEL_FACTOR)) + (target_speed * ACCEL_FACTOR)
            gyro_yaw = raw_telemetry[9] * (np.pi / 180.0)    
            
            distance_moved = estimated_speed * dt
            yaw_change = gyro_yaw * dt
            
            new_x = current_target_x * np.cos(yaw_change) - current_target_z * np.sin(yaw_change)
            new_z = current_target_x * np.sin(yaw_change) + current_target_z * np.cos(yaw_change)
            
            current_target_x = new_x
            current_target_z = new_z - distance_moved
            
            norm_telemetry = list(raw_telemetry)
            norm_telemetry[4] = np.clip(raw_telemetry[4] / 16.0, -1.0, 1.0)
            norm_telemetry[5] = np.clip(raw_telemetry[5] / 16.0, -1.0, 1.0)
            norm_telemetry[6] = np.clip(raw_telemetry[6] / 16.0, -1.0, 1.0)
            
            norm_telemetry[7] = np.clip(raw_telemetry[7] / 2000.0, -1.0, 1.0)
            norm_telemetry[8] = np.clip(raw_telemetry[8] / 2000.0, -1.0, 1.0)
            norm_telemetry[9] = np.clip(raw_telemetry[9] / 2000.0, -1.0, 1.0)
            
            norm_telemetry[10] = np.clip(raw_telemetry[10] / 4000.0, 0.0, 1.0)

            norm_target_x = float(current_target_x / MAX_ARENA_SIZE)
            norm_target_z = float(current_target_z / MAX_ARENA_SIZE)

            current_obs = np.array(norm_telemetry + yolo_obs + [norm_target_x, norm_target_z], dtype=np.float32)            
            
            frame_buffer.append(current_obs)
            stacked_obs = np.concatenate(list(frame_buffer), axis=0)

            throttle, steering, cam_pitch, cam_yaw = driver_agent.get_action(stacked_obs)

            # 2.0-Second Safety Kill Switch
            if current_time - last_command_time > MAX_EXEC_TIME:
                throttle = 0.0

            hardware.apply_actuators(throttle, steering, cam_pitch, cam_yaw)

            elapsed = time.perf_counter() - current_time
            sleep_time = loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Stopping...")
        is_running = False
    finally:
        print("Releasing actuators and camera safely.")
        hardware.close()
        cap.release()
        input_thread.join(timeout=1.0) 

if __name__ == "__main__":
    main()