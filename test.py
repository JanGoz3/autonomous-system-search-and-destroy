import asyncio
import websockets
import json
import threading
import math
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Buffers to hold position history (now tracking height 'y' as well)
x_history = deque(maxlen=2000)
z_history = deque(maxlen=2000)
y_history = deque(maxlen=2000)
current_yaw = 0.0

def quaternion_to_yaw(x, y, z, w):
    """Computes Yaw around the vertical axis."""
    siny_cosp = 2.0 * (w * y - x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

async def receive_pose(websocket):
    global current_yaw
    print("\n✅ Foxtail connected! Streaming 3D pose data...")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # 1. Parse position (strings to float)
                px = float(data['px'])
                pz = float(data['pz'])
                py = float(data['py']) # Vertical Height
                
                # 2. Parse quaternion rotation
                qx = float(data['x'])
                qy = float(data['y'])
                qz = float(data['z'])
                qw = float(data['w'])
                
                current_yaw = quaternion_to_yaw(qx, qy, qz, qw)
                
                x_history.append(px)
                z_history.append(pz)
                y_history.append(py)
                
            except (KeyError, ValueError, json.JSONDecodeError):
                pass
    except websockets.exceptions.ConnectionClosed:
        print("\n❌ Foxtail disconnected.")

async def run_server():
    # Modern websockets approach for Python 3.10+
    async with websockets.serve(receive_pose, "0.0.0.0", 5000):
        await asyncio.Future()

def start_ws_server():
    # Let asyncio handle creating and attaching the event loop automatically
    asyncio.run(run_server())

# ==========================================
# MATPLOTLIB 3D LIVE GRAPH SETUP
# ==========================================
fig = plt.figure(figsize=(10, 8))
# Initialize as a 3D projection
ax = fig.add_subplot(111, projection='3d')

# For 3D lines, initialize them empty
line, = ax.plot([], [], [], 'b-', linewidth=2, label="3D Path Traveled")
point, = ax.plot([], [], [], 'ro', markersize=6, label="Current Position")
heading_vector, = ax.plot([], [], [], 'r-', linewidth=2, label="Heading")

# Set initial limits
ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-1, 1)

ax.set_title("Foxtail Live ARKit 3D Odometry")
# We map px and pz to the floor plane, and py to the vertical Z-axis of the plot
ax.set_xlabel("Lateral (px - meters)")
ax.set_ylabel("Forward/Back (pz - meters)") 
ax.set_zlabel("Height (py - meters)")

ax.legend(loc='upper left')

def update(frame):
    if len(x_history) > 0:
        cx = x_history[-1]
        cz = z_history[-1]
        cy = y_history[-1]
        
        # update line using set_data_3d
        line.set_data_3d(list(x_history), list(z_history), list(y_history))
        point.set_data_3d([cx], [cz], [cy])
        
        # Draw a 0.3-meter heading vector pointing in current direction
        arrow_len = 0.3
        hx = cx + arrow_len * math.sin(current_yaw)
        hz = cz + arrow_len * math.cos(current_yaw)
        
        # Vector stays flat at current height (cy)
        heading_vector.set_data_3d([cx, hx], [cz, hz], [cy, cy])
        
        # Dynamic viewport scaling for all 3 axes
        x_min, x_max = min(x_history), max(x_history)
        z_min, z_max = min(z_history), max(z_history)
        y_min, y_max = min(y_history), max(y_history)
        
        ax.set_xlim3d(min(-2, x_min - 0.5), max(2, x_max + 0.5))
        ax.set_ylim3d(min(-2, z_min - 0.5), max(2, z_max + 0.5))
        ax.set_zlim3d(min(-1, y_min - 0.5), max(1, y_max + 0.5))
        
    return line, point, heading_vector

# 1. Start network listener in the background
ws_thread = threading.Thread(target=start_ws_server, daemon=True)
ws_thread.start()

print("🚀 WebSocket server active at ws://192.168.0.178:5000")

# 2. Start the live UI loop
ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()