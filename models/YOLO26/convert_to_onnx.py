from ultralytics import YOLO

model = YOLO("yolo_small.pt")

print('exporting to onnx...')
model.export(format='onnx', imgsz = 320, opset = 12)
print('export complete')