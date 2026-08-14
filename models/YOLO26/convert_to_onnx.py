from ultralytics import YOLO

model = YOLO("yolo_ours_v3.pt")

print('exporting to onnx...')
model.export(format='onnx', imgsz = 320, opset = 12)
print('export complete')