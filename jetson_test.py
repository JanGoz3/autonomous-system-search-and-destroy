import time
import threading
from collections import deque
import cv2
import numpy as np
import onnxruntime as ort

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
DRIVER_MODEL_PATH = "DriverNet.onnx"
YOLO_MODEL_PATH = "yolov4.onnx"
YOLO_INPUT_SIZE = (416, 416)

STACKED_VECTORS = 3
STATE_SPACE = 40
MAX_ARENA_SIZE = 20.0  
CONTROL_HZ = 20.0      

# Shared Variables for Thread Communication
current_target_x = 0.0
current_target_z = 0.0  
last_command_time = 0.0  # Tracks when the last terminal command was received
is_running = True

# ==========================================
# TERMINAL INPUT THREAD
# ==========================================
def terminal_input_thread():
    """
    Runs in the background. Waits for user input without blocking the car's control loop.
    """
    global current_target_x, current_target_z, is_running, last_command_time
    
    print("\n" + "="*50)
    print("🎯 TERMINAL TARGET CONTROL ACTIVE")
    print("Format: X, Z (e.g. '0.5, 2.0')")
    print("  X: Negative = Left, Positive = Right")
    print("  Z: Negative = Behind, Positive = Ahead")
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
                new_x = float(parts[0].strip())
                new_z = float(parts[1].strip())
                
                # Update the shared variables and reset the safety timer
                current_target_x = new_x
                current_target_z = new_z
                last_command_time = time.perf_counter()
                
                print(f">>> [ACCEPTED] Car routing to X:{current_target_x:.2f}m, Z:{current_target_z:.2f}m")
            else:
                print(">>> [ERROR] Invalid format. Type exactly like: 0.5, 2.0")
                
        except ValueError:
            print(">>> [ERROR] Please enter valid numbers.")
        except EOFError:
            break

# ==========================================
# HARDWARE INTERFACES & YOLO PROCESSOR (Placeholders)
# ==========================================
class CarHardware:
    def __init__(self): pass
    def get_telemetry(self): return [0.0] * 11
    def apply_actuators(self, throttle, steering, cam_pitch, cam_yaw): pass

class YoloProcessor:
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
    def process_frame(self, frame):
        return [0.0] * 27

# ==========================================
# MAIN INFERENCE LOOP
# ==========================================
def main():
    global is_running, current_target_x, current_target_z, last_command_time
    
    print("Initializing Models and Hardware...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    
    yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=providers)
    yolo_processor = YoloProcessor(yolo_session)

    driver_session = ort.InferenceSession(DRIVER_MODEL_PATH, providers=providers)
    driver_input_name = driver_session.get_inputs()[0].name

    hardware = CarHardware()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open physical camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_buffer = deque([np.zeros(STATE_SPACE, dtype=np.float32) for _ in range(STACKED_VECTORS)], maxlen=STACKED_VECTORS)
    loop_interval = 1.0 / CONTROL_HZ
    
    # Start the background thread for terminal input
    input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
    input_thread.start()

    # Initialize timing variables
    loop_last_time = time.perf_counter()
    last_command_time = time.perf_counter() # Prevent immediate timeout on startup

    try:
        while is_running:
            current_time = time.perf_counter()
            dt = current_time - loop_last_time
            loop_last_time = current_time

            # 1. Grab Physical Camera Frame
            ret, frame = cap.read()
            if not ret:
                continue

            # 2. Run YOLO Object Detection -> Returns 27 floats
            yolo_obs = yolo_processor.process_frame(frame)

            # 3. Read Hardware Sensors
            telemetry = hardware.get_telemetry()
            
            # ==========================================
            # ODOMETRY CALCULATION
            # ==========================================
            # Note: Update these indices based on where speed/yaw actually sit in your 11-float array
            current_speed = telemetry[0]  # Expected in meters per second (m/s)
            gyro_yaw = telemetry[9]       # Expected in radians per second (rad/s)
            
            distance_moved = current_speed * dt
            yaw_change = gyro_yaw * dt
            
            # Rotate target based on car's angular rotation
            new_x = current_target_x * np.cos(yaw_change) - current_target_z * np.sin(yaw_change)
            new_z = current_target_x * np.sin(yaw_change) + current_target_z * np.cos(yaw_change)
            
            # Translate target based on forward movement
            current_target_x = new_x
            current_target_z = new_z - distance_moved
            # ==========================================
            
            # Normalize for the neural network
            norm_target_x = float(current_target_x / MAX_ARENA_SIZE)
            norm_target_z = float(current_target_z / MAX_ARENA_SIZE)

            # 4. Assemble & Stack Frame
            current_obs = np.array(
                telemetry + [norm_target_x, norm_target_z] + yolo_obs, 
                dtype=np.float32
            )
            
            frame_buffer.append(current_obs)
            stacked_obs = np.concatenate(list(frame_buffer), axis=0)
            model_input = np.expand_dims(stacked_obs, axis=0)

            # 5. Run DriverNet PPO Inference
            outputs = driver_session.run(["continuous_actions"], {driver_input_name: model_input})
            throttle, steering, cam_pitch, cam_yaw = outputs[0][0]

            # ==========================================
            # SAFETY KILL SWITCH
            # ==========================================
            if current_time - last_command_time > 2.0:
                throttle = 0.0  # Cut the engine, let steering continue adjusting
            # ==========================================

            # 6. Apply Actions to Physical Servos
            hardware.apply_actuators(throttle, steering, cam_pitch, cam_yaw)

            # 7. Regulate Loop Frequency
            elapsed = time.perf_counter() - current_time
            sleep_time = loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Stopping...")
        is_running = False
    finally:
        print("Releasing actuators and camera safely.")
        hardware.apply_actuators(0.0, 0.0, 0.0, 0.0)
        cap.release()
        input_thread.join(timeout=1.0) 

if __name__ == "__main__":
    main()