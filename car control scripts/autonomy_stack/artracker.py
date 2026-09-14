import socket
import threading
import time
import csv
import os

class ARTracker:
    def __init__(self, port=5005):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.setblocking(False)
        
        self.x = 0.0
        self.y = 0.0 
        self.z = 0.0
        
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0
        
        self.last_update_time = 0.0
        self.running = True
        
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def _listen(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                decoded_data = data.decode('utf-8').strip()
                
                parts = decoded_data.split(',')
                if len(parts) == 7:
                    self.x = float(parts[0])
                    self.y = float(parts[1])
                    self.z = float(parts[2])
                    self.qx = float(parts[3])
                    self.qy = float(parts[4])
                    self.qz = float(parts[5])
                    self.qw = float(parts[6])
                    
                    self.last_update_time = time.time()
                    
            except BlockingIOError:
                time.sleep(0.005)
            except Exception as e:
                pass

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.sock.close()

if __name__ == "__main__":
    print("Waiting for UDP data on port 5005...")
    
    tracker = ARTracker(port=5005)
    log_filename = "ar_odometry_log.csv"
    
    try:
        with open(log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
            
            while True:
                current_time = time.time()
                time_since_last_packet = current_time - tracker.last_update_time
                
                writer.writerow([
                    current_time, 
                    tracker.x, tracker.y, tracker.z, 
                    tracker.qx, tracker.qy, tracker.qz, tracker.qw
                ])
                
                status = "OK" if time_since_last_packet < 0.2 else "LAG!"
                print(f"[{status}] X: {tracker.x:6.3f} | Y: {tracker.y:6.3f} | Z: {tracker.z:6.3f} | Delay: {time_since_last_packet:.3f}s")
                
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        print(f"\nStopped logging. Data saved to: {os.path.abspath(log_filename)}")
        tracker.release()