import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from QLed import QLed

from keras.models import load_model
from keras.preprocessing import image
import numpy as np


class CNN:
    def __init__(self):
        self.model = load_model("../CNN Transfer Learning/Model.model")
        self.categories = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]

    def predict(self,img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img = np.expand_dims(img, axis=0)
        result = self.model.predict([img])
        return self.categories[np.argmax(result[0])]


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = "Waste Sorting"
        self.setWindowTitle(self.title)
        self.initGui()
        self.show()
        self.CNN = CNN()
        self.imagePath = ""

    def initGui(self):
        mainLayout = QVBoxLayout()
        grid = QGridLayout()

        titleLabel = QLabel()
        titleLabel.setPixmap(QPixmap("./bins/title.png").scaled(500,100))

        hbox = QHBoxLayout()
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(["SVM", "CNN"])
        self.modelComboBox.activated.connect(self.handleCombo)
        self.featuresExtractionComboBox = QComboBox()
        self.featuresExtractionComboBox.addItems(["SIFT", "SURF", "ORB"])
        executeButton = QPushButton("Execute")
        executeButton.clicked.connect(self.handleExecute)

        hbox.addWidget(self.modelComboBox)
        hbox.addWidget(self.featuresExtractionComboBox)
        hbox.addWidget(executeButton)

        ImageVbox = QVBoxLayout()
        self.imageLabel = QLabel()
        importButton = QPushButton("Import")
        importButton.clicked.connect(self.openImage)

        ImageVbox.addWidget(self.imageLabel)
        ImageVbox.addWidget(importButton)

        binsHbox = QHBoxLayout()
        whiteBin = QLed(onColour=QLed.Grey, offColour=QLed.Red)
        blueBin = QLed(onColour=QLed.Blue, offColour=QLed.Red)
        yellowBin = QLed(onColour=QLed.Yellow, offColour=QLed.Red)
        orangeBin = QLed(onColour=QLed.Orange, offColour=QLed.Red)
        glassBin = QLed(onColour=QLed.Green, offColour=QLed.Red)

        whiteVbox = QVBoxLayout()
        blueVbox = QVBoxLayout()
        yellowVbox = QVBoxLayout()
        orangeVbox = QVBoxLayout()
        glassVbox = QVBoxLayout()

        whiteLabel = QLabel()
        whiteLabel.setPixmap(QPixmap("./bins/blanc.png").scaled(165,165))
        blueLabel = QLabel()
        blueLabel.setPixmap(QPixmap("./bins/bleu.png").scaled(165, 165))
        yellowLabel = QLabel()
        yellowLabel.setPixmap(QPixmap("./bins/jaune.png").scaled(165, 165))
        orangeLabel = QLabel()
        orangeLabel.setPixmap(QPixmap("./bins/orange.png").scaled(165, 165))
        glassLabel = QLabel()
        glassLabel.setPixmap(QPixmap("./bins/verre.jpg").scaled(165, 165))

        whiteVbox.addWidget(whiteLabel)
        blueVbox.addWidget(blueLabel)
        yellowVbox.addWidget(yellowLabel)
        orangeVbox.addWidget(orangeLabel)
        glassVbox.addWidget(glassLabel)

        self.bins = {"Blanc": whiteBin,
                "Bleu": blueBin,
                "Jaune": yellowBin,
                "Orange": orangeBin,
                "Verre": glassBin}

        whiteVbox.addWidget(whiteBin)
        blueVbox.addWidget(blueBin)
        yellowVbox.addWidget(yellowBin)
        orangeVbox.addWidget(orangeBin)
        glassVbox.addWidget(glassBin)

        binsHbox.addLayout(whiteVbox)
        binsHbox.addLayout(blueVbox)
        binsHbox.addLayout(yellowVbox)
        binsHbox.addLayout(orangeVbox)
        binsHbox.addLayout(glassVbox)

        grid.addLayout(hbox, 0, 1)
        grid.addLayout(ImageVbox, 1, 0)
        grid.addLayout(binsHbox, 1, 1)

        mainLayout.addWidget(titleLabel)
        mainLayout.addLayout(grid)
        self.setLayout(mainLayout)

    def openImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Choose an image', "", "Image files (*.jpg *.png)")

        pixmap = QPixmap(fileName).scaled(224,224)

        self.imagePath = fileName
        self.imageLabel.setPixmap(pixmap)
        self.resetBins()

    def handleCombo(self):
        self.resetBins()
        self.featuresExtractionComboBox.setEnabled(self.modelComboBox.currentText() == "SVM")

    def handleExecute(self):
        self.resetBins()
        if self.imagePath == "":
            QMessageBox.about(self, "Warning", "No image selected")
            return

        if self.modelComboBox.currentText() == "CNN":
            self.updateBins(self.CNN.predict(self.imagePath))

        else:
            pass

    def updateBins(self, prediction):
        self.resetBins()
        self.bins[prediction].value = True


    def resetBins(self):
        for bin in self.bins:
            self.bins[bin].value = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())