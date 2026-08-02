from ultralytics import YOLO

model = YOLO("yolo26m.pt")
results = model('orange_locker_corridor.png', conf=0.1)
for result in results:
    result.show()