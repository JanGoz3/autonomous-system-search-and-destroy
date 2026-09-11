import cv2
import numpy as np

YOLO_INPUT_SIZE = (320, 320)

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class YoloProcessor:
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        
        out_shape = session.get_outputs()[0].shape
        if len(out_shape) == 3:
            self.num_classes = out_shape[2] - 5
        else:
            self.num_classes = 4 

    def process_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, YOLO_INPUT_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) 
        img_input = np.expand_dims(img, axis=0) 

        yolo_outputs = self.session.run(None, {self.input_name: img_input})[0]
        predictions = np.squeeze(yolo_outputs, axis=0)

        boxes = []
        confidences = []
        class_ids = []
        confidence_threshold = 0.5
        
        for pred in predictions:
            conf = pred[4]
            if conf > confidence_threshold:
                class_id = int(round(pred[5])) 
                
                x_min, y_min, x_max, y_max = pred[0:4]
                
                w = x_max - x_min
                h = y_max - y_min
                
                boxes.append([float(x_min), float(y_min), float(w), float(h)])
                confidences.append(float(conf))
                class_ids.append(class_id)

        final_detections = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
            if len(indices) > 0:
                for i in indices.flatten():
                    x_min, y_min, w, h = boxes[i]
                    
                    x_max = x_min + w
                    y_max = y_min + h
                    cx = x_min + (w / 2)
                    cy = y_min + (h / 2)
                    area = w * h
                    
                    final_detections.append({
                        'x': cx, 'y': cy, 'w': w, 'h': h, 
                        'x_min': x_min, 'y_min': y_min,
                        'x_max': x_max, 'y_max': y_max,
                        'conf': confidences[i], 
                        'class_id': class_ids[i],
                        'area': area
                    })

        final_detections = sorted(final_detections, key=lambda d: d['area'], reverse=True)

        yolo_array = []
        for i in range(3):
            if i < len(final_detections):
                det = final_detections[i]
                
                class_one_hot = [0.0] * 4
                if det['class_id'] < 4:
                    class_one_hot[det['class_id']] = 1.0
                
                norm_x = (det['x_min'] - 160.0) / 160.0
                norm_y = (det['y_min'] - 160.0) / 160.0
                norm_w = det['x_max'] / 320.0
                norm_h = det['y_max'] / 320.0
                
                features = [norm_x, norm_y, norm_w, norm_h, det['conf']] + class_one_hot                
                yolo_array.extend(features)
            else:
                yolo_array.extend([0.0] * 9)

        return yolo_array, final_detections