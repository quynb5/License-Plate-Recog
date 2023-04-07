import cv2
import numpy as np
import torch
import yolov5
import math
from sort import *

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = []
with open('coco.names.txt', 'r') as f:
    classes = f.read().splitlines()
    # need_class = [0, 1, 2]

# model = yolov5.load('vehicle.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', 'vehicle.pt')
model.to(device)

vid_path = "E:/VHT_Intership/CV_AI/Dataset/Traffic_Video/british_highway_traffic.mp4"
# vid_path = "E:/VHT_Intership/CV_AI/Dataset/Traffic_Video/traffic_video.avi"


def check_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        colorsBRG = [x, y]
        print(colorsBRG)


tracker = Sort(max_age=20, min_hits=5)
cap = cv2.VideoCapture(vid_path)
counter1 = []
counter2 = []
while cv2.waitKey(1) < 0:
    detections = np.empty((0, 5))
    _, frame = cap.read()
    result = model(frame)

    line1 = cv2.line(frame, (870, 30), (1070, 30), (0, 0, 255), 2)
    line2 = cv2.line(frame, (430, 680), (980, 680), (0, 0, 255), 3)

    for index, row in result.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = math.ceil(float(row['confidence'])*100)
        class_name = str(row['name'])
        if confidence > 60:
            current_detection = np.array([x1, y1, x2, y2, confidence])
            detections = np.vstack((detections, current_detection))
    result = tracker.update(detections)
    for info in result:
        x1, y1, x2, y2, idx = info
        x1, y1, x2, y2, idx = int(x1), int(y1), int(x2), int(y2), int(idx)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(confidence), (math.ceil((x1 + x2) / 2), y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(idx), (x2, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        center_x, center_y = x1 + (x2-x1)//2, y1+(y2-y1)//2
        cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)

        if 25 < center_y < 35:
            line1 = cv2.line(frame, (870, 30), (1070, 30), (0, 255, 0), 2)
            if counter1.count(idx) == 0:
                counter1.append(idx)
        elif 670 < center_y < 680:
            line2 = cv2.line(frame, (430, 680), (980, 680), (0, 255, 0), 3)
            if counter2.count(idx) == 0:
                counter2.append(idx)
    cv2.putText(frame, str(len(counter1)), (1080, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(frame, str(len(counter2)), (990, 680), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
