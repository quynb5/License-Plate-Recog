import cv2
from ultralytics import YOLO
import numpy as np
import math
from sort import *

model = YOLO("yolov8s_best_51epochs.pt")
vid_path = "E:/VHT_Intership/CV_AI/Dataset/Traffic_Video/british_highway_traffic.mp4"

classes = ["car", "motorcycle"]
tracker = Sort(max_age=20, min_hits=3)
cap = cv2.VideoCapture(vid_path)

while True:
    detections = np.empty((0, 5))
    _, frame = cap.read()
    result = model(frame, stream=1)
    for info in result:
        parameters = info.boxes
        for details in parameters:
            x1, y1, x2, y2 = details.xyxy[0]
            cfd = details.conf[0]
            cfd = math.ceil(cfd*100)
            cls_detect = details.cls[0]
            class_name = classes[int(cls_detect)]

            if cfd > 60:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                # cv2.putText(frame, str(cfd), (math.ceil((x1+x2)/2), y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                current_detection = np.array([x1, y1, x2, y2, cfd])
                detections = np.vstack((detections, current_detection))
    result = tracker.update(detections)
    for info in result:
        x1, y1, x2, y2, idx = info
        x1, y1, x2, y2, idx = int(x1), int(y1), int(x2), int(y2), int(idx)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(cfd), (math.ceil((x1+x2)/2), y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(idx), (x2, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
