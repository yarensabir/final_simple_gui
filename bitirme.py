#!/usr/bin/python3
# -*- coding: utf-8 -*-

from bitirme_ui import *
import numpy as np
import time, os, cv2, sys

class image_signal(QObject):
    image = Signal(np.ndarray)

class worker(QRunnable):
    def __init__(self, input_folder=None):
        super(worker, self).__init__()
        self.folder = None
        self.file_format = None
        self.sig = image_signal()

    def check_input(self, input_folder):
        if os.path.isfile(self.folder):
            _, ext = os.path.splitext(self.folder)

            if ext == '.mp4' or ext == '.avi' or ext == '.jpg' or ext == '.jpeg' or ext == '.png' or ext == '.txt':
                self.format = ext
                self.folder = input_folder
                return True

            else:
                print("File format is not supported")
                print("Supported formats are 'mp4', 'avi', 'jpg', 'png' and 'jpeg'")
                return False

        elif os.path.isdir(self.folder):
            file_format = None
            out = list()
            for f in os.listdir(self.folder):
                name, ext = os.path.splitext(f)
                if not (ext == '.mp4' or ext == '.avi' or ext == '.jpg' or ext == '.jpeg' or ext == '.png'):
                    print("Unsupported file found in the folder")
                    print("file =", name + ext)
                    s = input("To skip this file press s, to interrupt procces press i")
                    if s != 's':
                        return False

                elif file_format is not None:
                    if ext == file_format:
                        out.append(name + ext)

                    else:
                        print("Found different file format")
                        print("exsisting file_format and count =", file_format, len(out))
                        print("file =", name + ext)
                        s = input("To skip this file press s, to remove other format prees r, to interrupt procces press i")
                        if s == "r":
                            out.clear()

                        elif s == "i":
                            return False

                else:
                    file_format = file_format
                    out.append(name)

            self.file_format = file_format
            self.folder = out
            return True
                
        else:
            print("Invalid Input")
            return False

    def run(self):
        if isinstance(self.folder, list):
            for file in self.folder:
                image = cv2.imread(file)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.sig.image.emit(image)

        elif self.file_format == ".mp4" or self.file_format == ".avi":
            cap = cv2.VideoCapture(self.folder + self.file_format)

            if not cap.isOpened():
                print("Error on opening video capture")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.sig.image.emit(frame)

                else:
                    break

        elif self.file_format == ".txt":
            with open(self.folder + self.file_format, 'r') as f:
                for line in f:
                    image = cv2.imread(line)
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.sig.image.emit(image)

        else:
            image = cv2.imread(self.folder + self.file_format)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sig.image.emit(image)

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Nesne Tespiti")

        self.ui.dosya_sec.clicked.connect(self.select_file)
        self.ui.klasor_sec.clicked.connect(self.select_folder)

        self.worker = worker()
        self.worker.setAutoDelete(False)

        self.worker.sig.image.connect(self.update_img)

        self.show()

    @Slot(np.ndarray)
    def update_img(self, img):
        pixmap = self.cv_to_pixmap(img)
        self.ui.widget.setPixmap(pixmap)

    def cv_to_pixmap(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        to_Qt_format = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(to_Qt_format)

    def select_file(self):
        dial = QFileDialog()
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        dial.setFileMode(QFileDialog.ExistingFiles)
        dial.setNameFilters(["Text files (*.txt)", "Images (*.png *.jpg)"])
        dial.selectNameFilter("Images (*.png *.jpg)")
        if dialog.exec_() == QFileDialog.Accept:
            file_name  = dial.selectedFiles()
            ret = self.worker.check_input(file_name)
            if ret:
                QThreadPool.globalInstance().start(self.worker)

    def select_folder(self):
        dial = QFileDialog()
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        dial.setFileMode(QFileDialog.Directory)
        dial.setOption(QFileDialog.ShowDirsOnly)
        if dial.exec_() == QFileDialog.Accept:
            folder_name  = dial.selectedFiles()
            ret = self.worker.check_input(folder_name)
            if ret:
                QThreadPool.globalInstance().start(self.worker)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())