import cv2
import torch

# load model
yolo_plate_recog = torch.hub.load('ultralytics/yolov5', 'custom', 'best_LPR_640_40epochs.pt')
yolo_plate_detect = torch.hub.load('ultralytics/yolov5', 'custom', 'plate_detect_256.pt')

# set model confidence threshold
# IoU default is 0.45
yolo_plate_recog.conf = 0.6


def plate_detection(image):
    results = yolo_plate_detect(image)
    coordinates = []
    x1, y1, x2, y2 = 0, 0, 0, 0
    for index, row in results.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        coordinates.append([x1, y1, x2, y2])

    # cropped_image = image[coordinates[0][1]:coordinates[0][3], coordinates[0][0]:coordinates[0][2]]
    # return cropped_image
    return coordinates


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


def check_lp_type(image):
    plate_type = 1
    h, w = image.shape[:2]
    if (w / h) < 2.5:
        plate_type = 2
    return plate_type


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


def plate_recognition(image):
    y_mid = int(image.shape[0] / 2)
    lp_result = yolo_plate_recog(image)
    lp_type = check_lp_type(image)
    lp_text = lp_export(lp_result, lp_type, y_mid)
    return lp_text
