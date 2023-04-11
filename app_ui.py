# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Admin\Downloads\Anaconda\Lib\site-packages\qt5_applications\Qt\bin\LicensePlate_Search.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(848, 452)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.fr_main_functions = QtWidgets.QFrame(self.centralwidget)
        self.fr_main_functions.setMaximumSize(QtCore.QSize(300, 16777215))
        self.fr_main_functions.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr_main_functions.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr_main_functions.setObjectName("fr_main_functions")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.fr_main_functions)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.fr_functions = QtWidgets.QFrame(self.fr_main_functions)
        self.fr_functions.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr_functions.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr_functions.setObjectName("fr_functions")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.fr_functions)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.fr_type_search = QtWidgets.QFrame(self.fr_functions)
        self.fr_type_search.setMaximumSize(QtCore.QSize(200, 16777215))
        self.fr_type_search.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr_type_search.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr_type_search.setObjectName("fr_type_search")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.fr_type_search)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.cbox_type_search = QtWidgets.QComboBox(self.fr_type_search)
        self.cbox_type_search.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cbox_type_search.setFont(font)
        self.cbox_type_search.setIconSize(QtCore.QSize(20, 20))
        self.cbox_type_search.setObjectName("cbox_type_search")
        self.cbox_type_search.addItem("")
        self.cbox_type_search.addItem("")
        self.verticalLayout_7.addWidget(self.cbox_type_search)
        self.verticalLayout_2.addWidget(self.fr_type_search)
        self.fr_file_path = QtWidgets.QFrame(self.fr_functions)
        self.fr_file_path.setMaximumSize(QtCore.QSize(400, 16777215))
        self.fr_file_path.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr_file_path.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr_file_path.setObjectName("fr_file_path")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.fr_file_path)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.le_file_path = QtWidgets.QLineEdit(self.fr_file_path)
        self.le_file_path.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.le_file_path.setFont(font)
        self.le_file_path.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.le_file_path.setText("")
        self.le_file_path.setObjectName("le_file_path")
        self.verticalLayout_4.addWidget(self.le_file_path, 0, QtCore.Qt.AlignBottom)
        self.verticalLayout_2.addWidget(self.fr_file_path, 0, QtCore.Qt.AlignTop)
        self.fr_choose_file = QtWidgets.QFrame(self.fr_functions)
        self.fr_choose_file.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr_choose_file.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr_choose_file.setObjectName("fr_choose_file")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.fr_choose_file)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.btn_choose_path = QtWidgets.QPushButton(self.fr_choose_file)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_choose_path.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("E:/Qt_Designer/folder_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_choose_path.setIcon(icon)
        self.btn_choose_path.setIconSize(QtCore.QSize(30, 30))
        self.btn_choose_path.setObjectName("btn_choose_path")
        self.verticalLayout_3.addWidget(self.btn_choose_path)
        self.verticalLayout_2.addWidget(self.fr_choose_file, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.fr_action = QtWidgets.QFrame(self.fr_functions)
        self.fr_action.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr_action.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr_action.setObjectName("fr_action")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.fr_action)
        self.horizontalLayout_2.setContentsMargins(0, 20, 0, 15)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_start = QtWidgets.QPushButton(self.fr_action)
        self.btn_start.setMaximumSize(QtCore.QSize(125, 16777215))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_start.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("E:/Qt_Designer/start_icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_start.setIcon(icon1)
        self.btn_start.setIconSize(QtCore.QSize(35, 35))
        self.btn_start.setObjectName("btn_start")
        self.horizontalLayout_2.addWidget(self.btn_start)
        self.btn_close = QtWidgets.QPushButton(self.fr_action)
        self.btn_close.setMaximumSize(QtCore.QSize(125, 16777215))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_close.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("E:/Qt_Designer/exit_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_close.setIcon(icon2)
        self.btn_close.setIconSize(QtCore.QSize(35, 35))
        self.btn_close.setObjectName("btn_close")
        self.horizontalLayout_2.addWidget(self.btn_close)
        self.verticalLayout_2.addWidget(self.fr_action)
        self.verticalLayout.addWidget(self.fr_functions)
        self.fr_results = QtWidgets.QFrame(self.fr_main_functions)
        self.fr_results.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr_results.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr_results.setObjectName("fr_results")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.fr_results)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.fr_output_img = QtWidgets.QFrame(self.fr_results)
        self.fr_output_img.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr_output_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr_output_img.setObjectName("fr_output_img")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.fr_output_img)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_license_plate = QtWidgets.QLabel(self.fr_output_img)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_license_plate.setFont(font)
        self.label_license_plate.setFrameShape(QtWidgets.QFrame.Box)
        self.label_license_plate.setScaledContents(True)
        self.label_license_plate.setAlignment(QtCore.Qt.AlignCenter)
        self.label_license_plate.setObjectName("label_license_plate")
        self.verticalLayout_8.addWidget(self.label_license_plate)
        self.verticalLayout_6.addWidget(self.fr_output_img)
        self.fr_output_text = QtWidgets.QFrame(self.fr_results)
        self.fr_output_text.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr_output_text.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr_output_text.setObjectName("fr_output_text")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.fr_output_text)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.le_text_plate = QtWidgets.QLineEdit(self.fr_output_text)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.le_text_plate.sizePolicy().hasHeightForWidth())
        self.le_text_plate.setSizePolicy(sizePolicy)
        self.le_text_plate.setMaximumSize(QtCore.QSize(300, 75))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.le_text_plate.setFont(font)
        self.le_text_plate.setText("")
        self.le_text_plate.setAlignment(QtCore.Qt.AlignCenter)
        self.le_text_plate.setObjectName("le_text_plate")
        self.verticalLayout_9.addWidget(self.le_text_plate)
        self.verticalLayout_6.addWidget(self.fr_output_text)
        self.verticalLayout.addWidget(self.fr_results)
        self.horizontalLayout.addWidget(self.fr_main_functions)
        self.fr_input_img = QtWidgets.QFrame(self.centralwidget)
        self.fr_input_img.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.fr_input_img.setFrameShadow(QtWidgets.QFrame.Plain)
        self.fr_input_img.setLineWidth(1)
        self.fr_input_img.setObjectName("fr_input_img")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.fr_input_img)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(5)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.le_notification = QtWidgets.QLineEdit(self.fr_input_img)
        self.le_notification.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.le_notification.setFont(font)
        self.le_notification.setObjectName("le_notification")
        self.verticalLayout_5.addWidget(self.le_notification)
        self.label_input_img = QtWidgets.QLabel(self.fr_input_img)
        self.label_input_img.setFrameShape(QtWidgets.QFrame.Box)
        self.label_input_img.setLineWidth(2)
        self.label_input_img.setText("")
        self.label_input_img.setScaledContents(True)
        self.label_input_img.setAlignment(QtCore.Qt.AlignCenter)
        self.label_input_img.setObjectName("label_input_img")
        self.verticalLayout_5.addWidget(self.label_input_img)
        self.horizontalLayout.addWidget(self.fr_input_img)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cbox_type_search.setItemText(0, _translate("MainWindow", "Image"))
        self.cbox_type_search.setItemText(1, _translate("MainWindow", "Video"))
        self.le_file_path.setPlaceholderText(_translate("MainWindow", "Path"))
        self.btn_choose_path.setText(_translate("MainWindow", "ChooseFile"))
        self.btn_start.setText(_translate("MainWindow", " Process"))
        self.btn_close.setText(_translate("MainWindow", "Quit"))
        self.label_license_plate.setText(_translate("MainWindow", "License Plate Crop"))
        self.le_notification.setText(_translate("MainWindow", "Notification:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
