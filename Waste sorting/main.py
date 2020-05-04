import sys

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from QLed import QLed
from SVM.Model import *
from keras.models import load_model
from keras.preprocessing import image


class CNN:
    def __init__(self):
        self.model = load_model("CNN Transfer Learning/Model.model")
        self.categories = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img = np.expand_dims(img, axis=0)
        result = self.model.predict([img])
        return self.categories[np.argmax(result[0])]


class SVM:
    def __init__(self):

        self.SVM_SIFT = ImageClassifierModel("SIFT", 1000)
        self.SVM_SURF = ImageClassifierModel("SURF", 1000)
        self.SVM_ORB = ImageClassifierModel("ORB", 1000)

        self.SVM_SIFT.load()
        self.SVM_SURF.load()
        self.SVM_ORB.load()

    def predict(self, img_path, model):

        if model == "SIFT":
            return self.SVM_SIFT.predict(img_path)
        elif model == "SURF":
            return self.SVM_SURF.predict(img_path)
        elif model == "ORB":
            return self.SVM_ORB.predict(img_path)


class App(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Waste Sorting")
        self.setWindowIcon(QIcon("./bins/icon.png"))
        self.CNN = CNN()
        self.SVM = SVM()
        self.imagePath = ""

        self.bins = {}
        self.imageLabel = QLabel()
        self.featuresExtractionComboBox = QComboBox()
        self.modelComboBox = QComboBox()
        self.init_gui()
        self.show()

    def init_gui(self):

        main_layout = QVBoxLayout()
        grid = QGridLayout()

        title_layout = self.init_title()
        bins_hbox = self.init_led_box()
        options_hbox = self.init_options()
        image_vbox = self.init_import_image()

        grid.addLayout(options_hbox, 0, 1)
        grid.addLayout(image_vbox, 1, 0)
        grid.addLayout(bins_hbox, 1, 1)

        main_layout.addLayout(title_layout)
        main_layout.addLayout(grid)
        self.setLayout(main_layout)

    def init_import_image(self):

        image_vbox = QVBoxLayout()
        self.imageLabel.setFixedSize(224, 224)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        import_button = QPushButton("Import")
        import_button.clicked.connect(self.open_image)
        image_vbox.addWidget(self.imageLabel)
        image_vbox.addWidget(import_button)

        return image_vbox

    def init_options(self):

        options_hbox = QHBoxLayout()
        self.modelComboBox.addItems(["SVM", "CNN"])
        self.modelComboBox.activated.connect(self.handle_combo)
        self.featuresExtractionComboBox.addItems(["SIFT", "SURF", "ORB"])
        self.featuresExtractionComboBox.activated.connect(self.handle_combo)
        execute_button = QPushButton("Execute")
        execute_button.clicked.connect(self.handle_execute)
        options_hbox.addWidget(self.modelComboBox)
        options_hbox.addWidget(self.featuresExtractionComboBox)
        options_hbox.addWidget(execute_button)

        return options_hbox

    def init_title(self):

        title_layout = QHBoxLayout()
        title_image_label = QLabel()
        title_image_label.setPixmap(QPixmap("./bins/title.png").scaled(300, 100, Qt.KeepAspectRatio))
        title_image_label.setAlignment(Qt.AlignCenter)
        title_text_label = QLabel()
        title_text_label.setPixmap(QPixmap("./bins/waste.jpg").scaled(500, 100, Qt.KeepAspectRatio))
        title_text_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_image_label)
        title_layout.addWidget(title_text_label)

        return title_layout

    def init_led_box(self):

        bins_hbox = QHBoxLayout()

        white_bin = QLed(onColour=QLed.Grey, offColour=QLed.Red)
        blue_bin = QLed(onColour=QLed.Blue, offColour=QLed.Red)
        yellow_bin = QLed(onColour=QLed.Yellow, offColour=QLed.Red)
        orange_bin = QLed(onColour=QLed.Orange, offColour=QLed.Red)
        glass_bin = QLed(onColour=QLed.Green, offColour=QLed.Red)

        self.bins = {"Blanc": white_bin, "Bleu": blue_bin, "Jaune": yellow_bin, "Orange": orange_bin,
                     "Verre": glass_bin}

        bins_list = [white_bin, blue_bin, yellow_bin, orange_bin, glass_bin]
        bins_images = ["blanc", "bleu", "jaune", "orange", "verre"]

        for i in range(5):
            label = QLabel()
            label.setPixmap(QPixmap("./bins/{}.png".format(bins_images[i])).scaled(165, 165))
            vbox = QVBoxLayout()
            vbox.addWidget(label)
            vbox.addWidget(bins_list[i])
            bins_hbox.addLayout(vbox)

        return bins_hbox

    def open_image(self):

        file_name, _ = QFileDialog.getOpenFileName(self, 'Choose an image', "", "Image files (*.jpg *.png)")

        pixmap = QPixmap(file_name).scaled(self.imageLabel.size(), Qt.KeepAspectRatio)

        self.imagePath = file_name
        self.imageLabel.setPixmap(pixmap)
        self.reset_bins()

    def handle_combo(self):
        self.reset_bins()
        self.featuresExtractionComboBox.setEnabled(self.modelComboBox.currentText() == "SVM")

    def handle_execute(self):
        self.reset_bins()
        if self.imagePath == "":
            QMessageBox.about(self, "Warning", "No image selected")
            return
        try:
            if self.modelComboBox.currentText() == "CNN":
                self.update_bins(self.CNN.predict(self.imagePath))

            else:
                self.update_bins(self.SVM.predict(self.imagePath, self.featuresExtractionComboBox.currentText()))
        except:
            QMessageBox.about(self, "Warning", "The image couldn't successfully be read")

    def update_bins(self, prediction):
        self.reset_bins()
        self.bins[prediction].value = True

    def reset_bins(self):
        for bin in self.bins:
            self.bins[bin].value = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
