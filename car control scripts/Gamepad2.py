import usb.core
import usb.util
import threading
import time
import struct

class GamepadState:
    def __init__(self):
        self.useRawValues = False
        
        self.axes = [0, 0, 0, 0]  # [left_x, left_y, right_x, right_y]
        self.triggers = [0, 0]  # [left_trigger, right_trigger]
        self.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # [X, Y, A, B, LB, RB, LS, RS, Back, Start]

        self.stickMaxValue = 32768
        self.triggersMaxValue = 255
        
        self.leftStickDeadzoneX = 0.15
        self.leftStickDeadzoneY = 0.15
        self.rightStickDeadzoneX = 0.15
        self.rightStickDeadzoneY = 0.15
        self.leftTriggerDeadzone = 0.00
        self.rightTriggerDeadzone = 0.00

        self.dev = None
        self.ep = None

        self.update_thread = threading.Thread(target=self.update_state_loop, daemon=True)
        self.update_thread.start()

    def connect_usb(self):
        self.dev = usb.core.find(idVendor=0x045e, idProduct=0x028e)
        if self.dev is None:
            return False

        try:
            if self.dev.is_kernel_driver_active(0):
                self.dev.detach_kernel_driver(0)
        except Exception:
            pass

        try:
            self.dev.set_configuration()
            cfg = self.dev.get_active_configuration()
            intf = cfg[(0,0)]
            self.ep = usb.util.find_descriptor(
                intf,
                custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
            )
            return self.ep is not None
        except Exception as e:
            return False

    def reset_gamepad(self):
        self.axes = [0, 0, 0, 0]
        self.triggers = [0, 0]
        self.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def normalize_axis_value(self, value, deadzone, maxValue):
        if self.useRawValues:
            return value / maxValue
        else:
            if value > deadzone:
                return round((value - deadzone) / (maxValue - deadzone), 4)
            elif value < -deadzone:
                return round((value + deadzone) / (maxValue - deadzone), 4)
            else:
                return 0

    def update_state_loop(self):
        while True:
            if self.dev is None or self.ep is None:
                if not self.connect_usb():
                    time.sleep(1)
                    continue

            try:
                data = self.dev.read(self.ep.bEndpointAddress, self.ep.wMaxPacketSize, timeout=100)
                self.process_usb_data(data)
            except usb.core.USBError as e:
                if "110" in str(e) or "10060" in str(e):
                    pass
                else:
                    self.dev = None
                    self.ep = None
                    self.reset_gamepad()

    def process_usb_data(self, data):
        if len(data) >= 14 and data[0] == 0x00:
            
            b1 = data[2]
            b2 = data[3]
            
            self.buttons[0] = 1 if (b2 & 0x40) else 0 # X
            self.buttons[1] = 1 if (b2 & 0x80) else 0 # Y
            self.buttons[2] = 1 if (b2 & 0x10) else 0 # A
            self.buttons[3] = 1 if (b2 & 0x20) else 0 # B
            self.buttons[4] = 1 if (b2 & 0x01) else 0 # LB
            self.buttons[5] = 1 if (b2 & 0x02) else 0 # RB
            self.buttons[6] = 1 if (b1 & 0x40) else 0 # LS
            self.buttons[7] = 1 if (b1 & 0x80) else 0 # RS
            self.buttons[8] = 1 if (b1 & 0x20) else 0 # Back
            self.buttons[9] = 1 if (b1 & 0x10) else 0 # Start

            raw_lt = data[4]
            raw_rt = data[5]
            
            raw_lx = struct.unpack('<h', data[6:8])[0]
            raw_ly = struct.unpack('<h', data[8:10])[0]
            raw_rx = struct.unpack('<h', data[10:12])[0]
            raw_ry = struct.unpack('<h', data[12:14])[0]

            self.triggers[0] = self.normalize_axis_value(raw_lt, self.leftTriggerDeadzone * 255, 255)
            self.triggers[1] = self.normalize_axis_value(raw_rt, self.rightTriggerDeadzone * 255, 255)
            
            self.axes[0] = self.normalize_axis_value(raw_lx, self.leftStickDeadzoneX * 32768, 32768)
            self.axes[1] = self.normalize_axis_value(raw_ly, self.leftStickDeadzoneY * 32768, 32768)
            self.axes[2] = self.normalize_axis_value(raw_rx, self.rightStickDeadzoneX * 32768, 32768)
            self.axes[3] = self.normalize_axis_value(raw_ry, self.rightStickDeadzoneY * 32768, 32768)

    def get_state(self):
        return self.axes, self.triggers, self.buttons

if __name__ == '__main__':
    gamepad_state = GamepadState()
    time_now = time.time()
    while True:
        time_from_start = time.time() - time_now
        axes, triggers, buttons = gamepad_state.get_state()
        print(f"Timestamp: {round(time_from_start, 2)}; Axes: {axes}, Triggers: {triggers}, Buttons: {buttons}")
        time.sleep(1/60)