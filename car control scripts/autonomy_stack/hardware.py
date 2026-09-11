import threading
import time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from socket_server import SerialCommunicator

class CarHardware:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        print(f"Connecting to Teensy on {port}...")
        self.communicator = SerialCommunicator(port, baudrate)
        self.telemetry_data = [0.0] * 11
        self.running = True
        
        self.rx_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.rx_thread.start()
        print("Hardware interface connected and listening.")

    def _listen_loop(self):
        buffer = ""
        raw_serial = getattr(self.communicator, 'ser', None) 
        
        if raw_serial is None:
            print("Warning: Could not access raw serial object for reading telemetry.")
            return

        while self.running:
            try:
                if raw_serial.in_waiting > 0:
                    chunk = raw_serial.read(raw_serial.in_waiting).decode('ascii', errors='ignore')
                    buffer += chunk
                    
                    while '<' in buffer:
                        start_idx = buffer.find('<')
                        if start_idx > 0:
                            buffer = buffer[start_idx:]
                            start_idx = 0
                            
                        end_idx = buffer.find('>')
                        if end_idx == -1:
                            break
                            
                        frame = buffer[start_idx : end_idx + 1]
                        buffer = buffer[end_idx + 1 :]
                        
                        self._parse_telemetry_frame(frame)            
            except Exception as e:
                print(f"Error parsing telemetry frame: {e}")
            
            time.sleep(0.005)

    def _parse_telemetry_frame(self, frame: str):
        try:
            if "state" in frame:
                parts = frame.strip("<>").split(";")
                if len(parts) >= 3:
                    values_str = parts[2]
                    values = [float(v) for v in values_str.split(',')]
                    if len(values) == 11:
                        self.telemetry_data = values
        except Exception:
            pass 

    def get_telemetry(self):
        return self.telemetry_data

    def apply_actuators(self, throttle, steering, cam_pitch, cam_yaw):
        steering_rounded = round(float(steering), 2)
        speed_rounded = round(float(throttle), 2)
        pitch_rounded = round(float(cam_pitch), 2)
        yaw_rounded = round(float(cam_yaw), 2)

        self.communicator.sendCommand("steering", [steering_rounded])
        self.communicator.sendCommand("speed", [speed_rounded])
        self.communicator.sendCommand("servos", [pitch_rounded, yaw_rounded])

    def close(self):
        self.running = False
        time.sleep(0.1)
        self.apply_actuators(0.0, 0.0, 0.0, 0.0)