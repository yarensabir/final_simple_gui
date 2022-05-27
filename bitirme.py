#!/usr/bin/python3
# -*- coding: utf-8 -*-

from bitirme_ui import *
import numpy as np
import time, os, cv2, sys
import darknet, random
import torch

class image_signal(QObject):
    image = Signal(np.ndarray)

class worker(QRunnable):
    def __init__(self, input_folder=None):
        super(worker, self).__init__()
        self.folder = None
        self.file_format = None
        self.sig = image_signal()
        self.stop_flag = False
        self.network_selection = "YoloV4"
        self.fbased = True
        self.network_size = 416

        self.yolov4_data_file = "/home/aye/yolov4/rb22/rb22.data"

    def check_input(self, input_folder):
        print("input =", input_folder)
        if os.path.isfile(input_folder):
            _, ext = os.path.splitext(input_folder)
            if ext == '.mp4' or ext == '.avi' or ext == '.jpg' or ext == '.jpeg' or ext == '.png' or ext == '.txt':
                self.file_format = ext
                self.folder = input_folder
                return True

            else:
                print("File format is not supported")
                print("Supported formats are 'mp4', 'avi', 'jpg', 'png' and 'jpeg'")
                return False

        elif os.path.isdir(input_folder):
            file_format = None
            out = list()
            skip_format = None
            for f in os.listdir(input_folder):
                name, ext = os.path.splitext(f)
                if ext == skip_format:
                    print("skipping file =", f)
                    pass

                elif not (ext == '.mp4' or ext == '.avi' or ext == '.jpg' or ext == '.jpeg' or ext == '.png'):
                    print("Unsupported file found in the folder")
                    print("file =", name + ext)
                    s = input("To skip this file press s, to skip this file type entairly press a, to interrupt procces press i = ")
                    if s == 'i':
                        return False

                    elif s == 'a':
                        skip_format = ext
                    
                    elif s == 's':
                        pass

                    else:
                        print("exiting")
                        return False

                elif file_format is not None:
                    if ext == file_format:
                        out.append(input_folder + "/" + name + ext)

                    else:
                        print("Found different file format")
                        print("exsisting file_format and count =", file_format, len(out))
                        print("file =", name + ext)
                        s = input("To skip this file press s, to remove other format prees r, to interrupt procces press i")
                        if s == "r":
                            out.clear()

                        elif s == "i":
                            print("exiting")
                            return False

                else:
                    file_format = ext
                    out.append(input_folder + "/" + name + ext)

            self.file_format = file_format
            self.folder = out
            print("exiting")
            return True
                
        else:
            print("Invalid Input =", input_folder)
            return False

    def run(self):
        if isinstance(self.folder, list):
            
            for file in self.folder:
                image = cv2.imread(file)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.sig.image.emit(image)

                if self.stop_flag:
                    break

        elif self.file_format == ".mp4" or self.file_format == ".avi":
            cap = cv2.VideoCapture(self.folder)

            if not cap.isOpened():
                print("Error on opening video capture")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.sig.image.emit(frame)

                else:
                    break

                if self.stop_flag:
                    break

        elif self.file_format == ".txt":
            with open(self.folder + self.file_format, 'r') as f:
                for line in f:
                    image = cv2.imread(line)
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.sig.image.emit(image)

                    if self.stop_flag:
                        break

        else:
            image = cv2.imread(self.folder)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sig.image.emit(image)

    def run_on_network(self, image):
        if self.network_selection == "YoloV4":
            return self.run_on_yolov4(image)

        elif self.network_selection == "YoloV5":
            return self.run_on_yolov5(image)
        
        elif self.network_selection == "YOLOR":
            return self.run_on_yolor(image)

    def del_network(self):
        if self.network_selection == "YoloV4":
            darknet.free_network_ptr(self.network)

        elif self.network_selection == "YoloV5":
            del self.network

        elif self.network_selection == "YOLOR":
            pass

    def load_network(self):
        if self.network_selection == "YoloV4":
            config_file = "/home/aye/bitirme_networks/" + "fbased/" if self.fbased else "cbased/" + "yolov4/" + str(self.size) + "/bitirme.cfg"
            weights_file = "/home/aye/bitirme_networks/" + "fbased/" if self.fbased else "cbased/" + "yolov4/" + str(self.size) + "/best.weights"
            self.network, self.class_names, self.colors = darknet.load_network(config_file, self.yolov4_data_file, weights_file)

        elif self.network_selection == "YoloV5":
            pth = "/home/aye/bitirme_networks/" + "fbased/" if self.fbased else "cbased/" + "yolov5/" + str(self.size) + "/best.pt"
            self.network = torch.hub.load("ultralytics/yolov5", 'custom', path=pth, device='cuda')

        elif self.network_selection == "YOLOR":
            pass

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Nesne Tespiti")

        self.ui.dosya_sec.clicked.connect(self.select_file)
        self.ui.klasor_sec.clicked.connect(self.select_folder)
        self.ui.durdur.clicked.connect(self.set_stop_flag)
        self.ui.comboBox.indexChanged(self.get_selected_network)

        self.worker = worker()
        self.worker.setAutoDelete(False)

        self.worker.sig.image.connect(self.update_img)

        pmap = QPixmap("output-onlinepngtools (7).png")
        self.ui.label.setPixmap(pmap)

        self.show()

    def get_selected_network(self):
        nt. type, size = self.ui.comboBox.currentText().split("-")
        self.worker.del_network()
        self.worker.network_selection = nt
        self.worker.fbased = True if type == "FB" else False
        self.worker.network_size = int(size)
        self.worker.load_network()

    def set_stop_flag(self):
        self.worker.stop_flag = True

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
        if dial.exec_():
            file_name  = dial.selectedFiles()
            ret = self.worker.check_input(file_name[0])
            if ret:
                self.worker.stop_flag = False
                QThreadPool.globalInstance().start(self.worker)
        
    def select_folder(self):
        dial = QFileDialog()
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        dial.setFileMode(QFileDialog.Directory)
        dial.setOption(QFileDialog.ShowDirsOnly)
        if dial.exec_():
            folder_name  = dial.selectedFiles()
            ret = self.worker.check_input(folder_name[0])
            if ret:
                self.worker.stop_flag = False
                QThreadPool.globalInstance().start(self.worker)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())