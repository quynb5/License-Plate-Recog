from PIL import Image
from pathlib import Path
import os

# Open the image
root = "E:/VHT_Intership/CV_AI/Dataset/LicensePlate/GreenParking"
txt_path = "E:/VHT_Intership/CV_AI/Dataset/LicensePlate/GreenParking/location.txt"
dest_root = "E:/VHT_Intership/CV_AI/Dataset/LicensePlate/GreenParking_Croped"

with open(txt_path) as f:
    for i in range(200):
        infors = f.readline().rstrip().split(" ")
        img_file = infors[0]
        img_path = root + "/" + img_file
        coordinates = [eval(pixel) for pixel in infors[2:]]
        image = Image.open(img_path)
        coordinates[2] += coordinates[0]
        coordinates[3] += coordinates[1]

        # Crop the image using the coordinates
        cropped_image = image.crop(coordinates)
        # Save the cropped image
        image_name = dest_root + "/" + str(i) + ".jpg"
        path = Path(image_name)
        cropped_image.save(image_name)
