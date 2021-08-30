import os

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QImageReader
from PyQt5.QtWidgets import (QWidget,
                             QPushButton, QFileDialog, QLabel, QMessageBox)
from processor import *
from model.frcnn import FRCNN

class myWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.img_label = QLabel(self)
        self.cwd = os.getcwd()
        self.img = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('垃圾检测系统')
        self.img_label.setGeometry(100, 100, 600, 600)
        self.resize(800, 800)
        sel_btn = QPushButton("加载图片", self)
        sel_btn.resize(sel_btn.sizeHint())
        sel_btn.clicked.connect(self.getimage)
        sel_btn.move(255, 720)
        test_btn = QPushButton("检测垃圾", self)
        test_btn.resize(sel_btn.sizeHint())
        test_btn.clicked.connect(self.test)
        test_btn.move(455, 720)

    def getimage(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "选取文件", self.cwd,
                                                         "Image files (*.jpg *.gif *.png *.jpeg)")
        if filename == "":
            QMessageBox.information(self, "Warning", "图片读取失败")
            return
        img = QImageReader(filename)
        scale = 800 / img.size().width()
        height = int(img.size().height() * scale)
        img.setScaledSize(QSize(800, height))
        img = img.read()
        self.showimage(img)
        self.img = Image.open(filename)

    def showimage(self, image):
        pix = QPixmap(image)
        self.img_label.setPixmap(pix)

    def test(self):
        if self.img == None:
            QMessageBox.information(self, "Warning", "图片加载到模型失败", QMessageBox.Yes)
            return
        frcnn = FRCNN()
        output = frcnn.detect_image(self.img)
        plt.imshow(output)
        plt.show()