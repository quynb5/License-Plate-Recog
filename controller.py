# import os
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from app_ui import Ui_MainWindow
from license_plate import plate_recognition, plate_detection


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.img_path = self.ui.le_file_path.text()
        self.type_search_id = self.ui.cbox_type_search.currentIndex()
        self.ui.btn_choose_path.clicked.connect(self.get_file_path)
        self.ui.btn_choose_path.clicked.connect(self.insert_image)
        self.ui.btn_start.clicked.connect(self.plate_recognize)
        self.ui.btn_close.clicked.connect(self.close)
        self.ui.label_input_img.mousePressEvent = self.check_positon

    # Choose folder by button
    def get_file_path(self):
        self.type_search_id = self.ui.cbox_type_search.currentIndex()
        file_path = None
        if self.type_search_id == 0:
            file_path, _ = QtWidgets.QFileDialog().getOpenFileName(self, "Lựa chọn file ảnh", directory="E:",
                                                                   filter='Images (*.png *.jpg)')
        elif self.type_search_id == 1:
            file_path, _ = QtWidgets.QFileDialog().getOpenFileName(self, "Choose Video File", directory="E:",
                                                                   filter='Video (*.mp4)')
        self.ui.le_file_path.setText(file_path)

    # Insert image into label
    def insert_image(self):
        self.ui.label_input_img.clear()
        self.img_path = self.ui.le_file_path.text()
        image = cv2.imread(self.img_path)
        image = cv2.resize(image, (200, 200))
        pix_img = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                               QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.label_input_img.setPixmap(QtGui.QPixmap.fromImage(pix_img))

    # Recognize license plate and put image, text into QLabel, QLineEdit
    def plate_recognize(self):
        self.ui.le_text_plate.clear()
        self.ui.label_license_plate.clear()
        path = self.ui.le_file_path.text()
        vehicle_img = cv2.imread(path)
        coordinates = plate_recognition(vehicle_img)

        # Set plate image into Label
        # plate_img = plate_detection(vehicle_img)

        # Fix start here
        coordinates = plate_recognition(vehicle_img)
        cropped_img = image[coordinates[0][1]:coordinates[0][3], coordinates[0][0]:coordinates[0][2]]
        image = cv2.resize(cropped_img, (60, 60))
        # Fix end here

        # image = cv2.resize(plate_img, (60, 60))
        pix_img = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                               QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.label_license_plate.setPixmap(QtGui.QPixmap.fromImage(pix_img))

        # Set license plate digits into LineEdit
        plate_text = plate_recognition(plate_img)
        self.ui.le_text_plate.setText(plate_text)

    # Check mouse press on license plate regions
    def check_positon(self, event):
        img_size = self.ui.label_input_img.size()

        # QRect(x1, y1, x2, y2)
        plate_rect = QtCore.QRect(0, 0, 300, 100)
        print(plate_rect.contains(event.pos()))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
