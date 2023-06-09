import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from app_ui import Ui_MainWindow
from license_plate import image_process, vehicle_detect


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.file_path = ""
        self.input_image = None
        self.infors = dict()
        self.type_search_id = self.ui.cbox_type_search.currentIndex()
        self.ui.btn_choose_path.clicked.connect(self.get_file_path)
        self.ui.btn_choose_path.clicked.connect(self.insert_image)
        self.ui.btn_start.clicked.connect(self.plate_recognize)
        self.ui.btn_close.clicked.connect(self.close)
        self.ui.label_input_img.mousePressEvent = self.check_position

    # Choose folder by button
    def get_file_path(self):
        self.ui.le_file_path.clear()
        self.type_search_id = self.ui.cbox_type_search.currentIndex()
        if self.type_search_id == 0:
            self.file_path, _ = QtWidgets.QFileDialog().getOpenFileName(self, "Lựa chọn file ảnh",
                                                                        directory="E:/VHT_Intership/CV_AI/Dataset/"
                                                                                  "LicensePlate/GreenParking",
                                                                        filter='Images (*.png *.jpg)')
        elif self.type_search_id == 1:
            self.file_path, _ = QtWidgets.QFileDialog().getOpenFileName(self, "Choose Video File", directory="E:",
                                                                        filter='Video (*.mp4)')
        self.ui.le_file_path.setText(self.file_path)

    # Insert image into label
    def insert_image(self):
        self.ui.label_input_img.clear()
        self.input_image = cv2.imread(self.file_path)
        # Convert RGB image to Pixmap for display on QLabel
        image = cv2.resize(self.input_image, (400, 400))
        pix_img = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.label_input_img.setPixmap(QtGui.QPixmap.fromImage(pix_img))

        # If choose new image, reset infors to None
        self.infors.clear()

    # Process input image and return informations
    def plate_recognize(self):
        self.infors = image_process(self.input_image)
        self.display_detect_image()
        self.display_notification("Image processing successfully")

    # Check mouse press on license plate regions
    def check_position(self, event):
        if bool(self.infors):
            plates_text = list(self.infors.keys())
            plates_infor = list(self.infors.values())
            # new_coordinates = self.scale_coordinates(coordinates)

            flag = "Choose False"
            for i, plate_info in enumerate(plates_infor):
                vehicle_name, vehicle_coordinate, plate_coordinate = plate_info
                new_vehicle_coordinate = self.scale_coordinates(vehicle_coordinate)
                # Create a empty rectangle and set coordinates based on license plate
                plate_rect = QtCore.QRect()
                plate_rect.setCoords(*new_vehicle_coordinate)
                # If click on rectangle
                if plate_rect.contains(event.pos()):
                    flag = "Choose True"
                    # Display license plate and digits of it
                    self.display_plate_image(plate_coordinate)
                    self.display_plate_text(plates_text[i])
            self.display_notification(flag)
            if flag == "Choose False":
                # Display Unknown
                self.ui.label_license_plate.clear()
                self.display_plate_text("Unknown")

        else:
            self.display_notification("Not have plate information from input image")

    # Display license plate image user have choose
    def display_plate_image(self, coordinate):
        self.ui.label_license_plate.clear()
        x1, y1, x2, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
        plate_image = self.input_image[y1:y2, x1:x2]
        # Set plate image into Label
        plate_image = cv2.resize(plate_image, (100, 100))
        pix_plate_img = QtGui.QImage(plate_image.data, plate_image.shape[1], plate_image.shape[0],
                                     QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.label_license_plate.setPixmap(QtGui.QPixmap.fromImage(pix_plate_img))

    # Display input image after detect vehicle
    def display_detect_image(self):
        display_image = self.input_image.copy()
        vehicle_info = vehicle_detect(self.input_image)
        vehicle_coordinates = [info[1] for info in vehicle_info]
        for coordinate in vehicle_coordinates:
            x1, y1, x2, y2 = coordinate
            display_image = cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        # Convert RGB image to Pixmap for display on QLabel
        image = cv2.resize(display_image, (400, 400))
        pix_img = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.label_input_img.clear()
        self.ui.label_input_img.setPixmap(QtGui.QPixmap.fromImage(pix_img))

    # Display license plate digits
    def display_plate_text(self, plate_text):
        self.ui.le_text_plate.clear()
        self.ui.le_text_plate.setText(plate_text)

    # Scale plate coordinates to fit Qlabel size
    def scale_coordinates(self, old_coordinate):
        new_coordinate = [None]*4
        h, w, _ = self.input_image.shape
        new_size = self.ui.label_input_img.size()
        new_h, new_w = new_size.height(), new_size.width()
        scale_x, scale_y = new_w / w, new_h / h
        new_coordinate[0], new_coordinate[2] = int(old_coordinate[0] * scale_x), int(old_coordinate[2] * scale_x)
        new_coordinate[1], new_coordinate[3] = int(old_coordinate[1] * scale_y), int(old_coordinate[3] * scale_y)

        return new_coordinate

    # Display notifications on QLineEdit le_notification
    def display_notification(self, notification):
        self.ui.le_notification.clear()
        noti_text = "Notification: " + notification
        self.ui.le_notification.setText(noti_text)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # style_sheet = """
    #     QWidget{
    #         background: #262D37
    #     }
    #     QLabel{
    #         color: #fff
    #     }
    #     QLabel#label_input_img, QLabel#label_license_plate{
    #         border: 1px solid #fff;
    #         border-radius: 8px
    #         padding: 2px
    #     }
    # """
    # app.setStyleSheet(style_sheet)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
