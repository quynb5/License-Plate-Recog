import cv2
import torch
from ultralytics import YOLO

# load model
yolo_plate_recog = torch.hub.load('ultralytics/yolov5', 'custom', 'best_LPR_640_40epochs.pt')
yolo_plate_detect = torch.hub.load('ultralytics/yolov5', 'custom', 'plate_detect_256.pt')
yolo_vehicle_detect = YOLO("yolov8s_best_51epochs.pt")

# set model confidence threshold
# IoU default is 0.45
yolo_plate_recog.conf = 0.6
yolo_plate_detect.conf = 0.6

classes = ["car", "motorbike"]


# Detect vehicles and get information (class_name, coordinate)
def vehicle_detect(image):
    vehicles_info = list()
    results = yolo_vehicle_detect(image, conf=0.6)
    for info in results:
        for details in info.boxes:
            x1, y1, x2, y2 = details.xyxy[0]
            vehicle_coordinate = [int(x1), int(y1), int(x2), int(y2)]
            # cfd = math.ceil(details.conf[0] * 100)
            class_name = classes[int(details.cls[0])]
            vehicles_info.append([class_name, vehicle_coordinate])
    return vehicles_info


# Check type of license plate (1 or 2 line) (Not use in this code)
def check_type(results):
    plate_type = 1
    list_x = []
    length_digit = 0
    for index, row in results.pandas().xyxy[0].iterrows():
        x1, x2 = int(row['xmin']), int(row['xmax'])
        list_x.append(x1)
        list_x.append(x2)
        length_digit += (x2 - x1)
    list_x.sort()
    length_plate = list_x[-1] - list_x[0]
    print(length_plate / length_digit)
    if length_plate / length_digit < 0.9:
        plate_type = 2
    return plate_type


# Check type of license plate (1 or 2 line) (Use in this code)
def check_lp_type(image):
    plate_type = 1
    h, w = image.shape[:2]
    if (w / h) < 2.5:
        plate_type = 2
    return plate_type


# Export digits of license plate, return plate text
def lp_export(results, plate_type, y_middle):
    plate_text = ""
    list_center = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        digit = row['name']
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        list_center.append((x_center, y_center, digit))
    if plate_type == 1:
        list_center.sort(key=lambda a: a[0])
        for i in range(len(list_center)):
            plate_text += list_center[i][2]
            if i == 2:
                plate_text += "-"
    else:
        top_center, down_center = [], []
        for x in list_center:
            top_center.append(x) if x[1] < y_middle else down_center.append(x)
        top_center.sort(key=lambda a: a[0])
        down_center.sort(key=lambda a: a[0])
        # list_center
        for i in range(len(top_center)):
            plate_text += top_center[i][2]
            if len(top_center) == 4 and i == 1:
                plate_text += "-"
        plate_text += " "
        for i in range(len(down_center)):
            plate_text += down_center[i][2]
    return plate_text


# Recognize digits in license plate
def plate_recognition(image):
    y_mid = int(image.shape[0] / 2)
    lp_result = yolo_plate_recog(image)
    lp_type = check_lp_type(image)
    lp_text = lp_export(lp_result, lp_type, y_mid)
    return lp_text


# Detect plate and get information (coordinate, digits)
def plate_process(vehicle_image, vehicle_coordinate):
    X1, Y1, X2, Y2 = vehicle_coordinate
    results = yolo_plate_detect(vehicle_image)
    coordinate = [None]*4
    plate_text = ""
    # if results.pandas().xyxy[0].empty, loop "for" not active, coordinate is None
    for index, row in results.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        coordinate = [X1+x1, Y1+y1, X1+x2, Y1+y2]
        cropped_image = vehicle_image[y1:y2, x1:x2]
        plate_text = plate_recognition(cropped_image)
    return coordinate, plate_text


# Process image (detect vehicle, detect license plate and recognize it) and return dictionary information
# Include key is license plate text, values are vehicle name, vehicle coordinate and plate coordinate
def image_process(input_image):
    image_info = dict()
    vehicle_infors = vehicle_detect(input_image)
    for i, info in enumerate(vehicle_infors):
        class_name, vehicle_coordinate = info
        x1, y1, x2, y2 = vehicle_coordinate
        vehicle_image = input_image[y1:y2, x1:x2]
        plate_coordinate, plate_text = plate_process(vehicle_image, vehicle_coordinate)
        # Check can detect license plate from vehicle image
        # If any plate_coordinate is not None, mean can detect license plate
        if any(plate_coordinate):
            vehicle_infors[i].append(plate_coordinate)
            image_info.update({plate_text: vehicle_infors[i]})
    return image_info


