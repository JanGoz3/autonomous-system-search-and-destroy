import serial
import time
import threading
import datetime

class TeensyLogger:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200, log_filename=None):
        print(f"Próba otwarcia połączenia nasłuchowego na porcie {port}...")
        self.ser = serial.Serial(port, baudrate, timeout=0.05)
        self.running = True

        if log_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"teensy_log_{timestamp}.txt"

        self.log_file = open(log_filename, "a", encoding="utf-8")
        self.log_lock = threading.Lock()

        print(f"Połączono. Logowanie do pliku: {log_filename}")

        self.rx_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.rx_thread.start()
        print("Nasłuch w tle aktywny\n")

    def _log_raw_frame(self, frame: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {frame}"
        
        with self.log_lock:
            self.log_file.write(log_line + "\n")
            self.log_file.flush()
        
        print(log_line)

    def _listen_loop(self):
        buffer = ""
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    chunk = self.ser.read(self.ser.in_waiting).decode('ascii', errors='ignore')
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
                        
                        self._log_raw_frame(frame)
                        
            except Exception as e:
                print(f"Wyjątek podczas odczytu: {e}")
            
            time.sleep(0.005)

    def close(self):
        self.running = False
        time.sleep(0.1)
        if self.ser.is_open:
            self.ser.close()
        print("\nPołączenie zamknięte.")
        with self.log_lock:
            self.log_file.close()
            print("Plik logowania poprawnie zamknięty.")


if __name__ == '__main__':
    try:
        logger = TeensyLogger(port="/dev/ttyACM0", baudrate=115200)
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika.")
    except serial.SerialException as e:
        print(f"\nBłąd portu szeregowego: {e}")
    finally:
        if 'logger' in locals():
            logger.close()