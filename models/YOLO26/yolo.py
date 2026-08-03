import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

results = model('asd.png')

for result in results:

    annotated_frame = result.plot(boxes=True, conf=True)
    cv2.imshow("Podglad celow - Tylko ramki", annotated_frame)
    
cv2.waitKey(0)
cv2.destroyAllWindows()