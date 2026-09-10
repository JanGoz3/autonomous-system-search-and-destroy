import cv2
from ultralytics import YOLO

model = YOLO("yolo_ours_v4.pt")

img = cv2.imread('test_images/bbb.png')
#img = cv2.resize(img, (640, 640))
results = model(img)

for result in results:

    annotated_frame = result.plot(boxes=True, conf=True)
    cv2.namedWindow('display', cv2.WINDOW_NORMAL)
    cv2.imshow('display', annotated_frame)
    cv2.moveWindow('display', 50, 50)
    
cv2.waitKey(0)
cv2.destroyAllWindows()