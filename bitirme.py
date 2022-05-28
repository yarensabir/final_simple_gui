#!/usr/bin/python3
# -*- coding: utf-8 -*-

from bitirme_ui import *
from yolor import *
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
        self.network_type = ""
        self.fbased = True
        self.network_size = 416

        self.yolov4_data_file = "/home/aye/yolov4/rb22/rb22.data"
        self.device = torch.device('cuda')

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
            del self.network

    def load_network(self):
        if self.network_selection == "YoloV4":
            config_file = "/home/aye/bitirme_networks/" + "fbased/" if self.fbased else "cbased/" + "yolov4/" + str(self.size) + "/bitirme.cfg"
            weights_file = "/home/aye/bitirme_networks/" + "fbased/" if self.fbased else "cbased/" + "yolov4/" + str(self.size) + "/best.weights"
            self.network, self.class_names, self.colors = darknet.load_network(config_file, self.yolov4_data_file, weights_file)

        elif self.network_selection == "YoloV5":
            pth = "/home/aye/bitirme_networks/" + "fbased/" if self.fbased else "cbased/" + "yolov5/" + str(self.network_type) + str(self.size) + "/best.pt"
            self.network = torch.hub.load("ultralytics/yolov5", 'custom', path=pth, device='cuda')

        elif self.network_selection == "YOLOR":
            pth = "/home/aye/bitirme_networks/" + "fbased/" if self.fbased else "cbased/" + "yolor/" + str(self.network_type) + str(self.size) + "/best.pt"
            self.network = load_yolor(pth, device=self.device)
            self.network = self.network.half()
            self.class_names = self.model.module.names if hasattr(model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]

    def run_on_yolov4(self, img):
        prev_time = time.time()

        drk_image = darknet.make_image(self.size, self.size, 3)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_resize = cv2,resize(img_rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(drk_image, im_resize.tobytes())

        detections = darknet.detect_image(self.network, self.class_names, drk_image)
        darknet.free_image(drk_image)

        return darknet.draw_boxes(detections, im_resize, self.colors)

    def run_on_yolov5(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.network(img_rgb, size=self.size)

        results.print()

    def run_on_yolor(self, img):
        orj_img = img.copy()

        img = letterbox(img, self.size, 64)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half()

        img /= 255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.network(img)[0]
        pred = non_max_suppression(pred)

        for i, det in enumerate(pred):
            gn = torch.tensor(orj_img.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orj_img.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (self.class_names[int(cls)], conf)
                    plot_one_box(xyxy, orj_img, label=label, color=self.colors[int(cls)], line_thickness=3)
        
        return orj_img

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
        if len(nt.split("_")) > 1:
            ns, nt = nt.split("_")
            self.worker.network_selection = ns
            self.worker.network_type = nt

        else:
            self.worker.network_selection = nt
            self.worker.network_type = ""

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