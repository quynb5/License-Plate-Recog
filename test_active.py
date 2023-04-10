import cv2
import torch
yolo_plate_detect = torch.hub.load('ultralytics/yolov5', 'custom', 'plate_detect_256.pt')
yolo_plate_detect.conf = 0.9
yolo_plate_recog = torch.hub.load('ultralytics/yolov5', 'custom', 'best_LPR_640_40epochs.pt')
yolo_plate_recog.conf = 0.1

path = "E:/VHT_Intership/CV_AI/Dataset/LicensePlate/GreenParking/0000_00532_b.jpg"
img = cv2.imread(path)
print(type(img))
results = yolo_plate_detect(img)
x1, y1, x2, y2 = 0, 0, 0, 0
for index, row in results.pandas().xyxy[0].iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
image = img[y1:y2, x1:x2]
print(type(image))
# # cv2.imshow("Image", image)
# results2 = yolo_plate_recog(image)
# print(results2.pandas().xyxy[0])
# results2.show()
# image = cv2.resize(img, (500, 320))
# cv2.imshow("Image", img)
# cv2.waitKey(0)
