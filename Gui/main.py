import sys
sys.path.append('../SVM code/')


from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from QLed import QLed

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

from Model import *
import joblib

class CNN:
    def __init__(self):
        self.model = load_model("../CNN Transfer Learning/Model.model")
        self.categories = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]

    def predict(self,img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img = np.expand_dims(img, axis=0)
        result = self.model.predict([img])
        return self.categories[np.argmax(result[0])]

class SVM:
    def __init__(self):

        self.SIFT_Cluster = ClusterModel()
        self.SIFT_Cluster.load_model('../SVM code/cluster-model/1000-SIFT.pkl')
        self.SURF_Cluster = ClusterModel()
        self.SURF_Cluster.load_model('../SVM code/cluster-model/1000-SURF.pkl')
        self.ORB_Cluster = ClusterModel()
        self.ORB_Cluster.load_model('../SVM code/cluster-model/1000-ORB.pkl')

        self.SVM_SIFT = ImageClassifierModel()
        self.SVM_SIFT.load_model('../SVM code/classification-model/1000-SVM-SIFT.pkl')
        self.SVM_SURF = ImageClassifierModel()
        self.SVM_SURF.load_model('../SVM code/classification-model/1000-SVM-SURF.pkl')
        self.SVM_ORB = ImageClassifierModel()
        self.SVM_ORB.load_model('../SVM code/classification-model/1000-SVM-ORB.pkl')

    def predict(self, img_path, model):
        image_desc = self.features_extraction(img_path, model, resize = (384, 512))
        if model == "SIFT":
            cluster_model = self.SIFT_Cluster
            classifier = self.SVM_SIFT
        elif model == "SURF":
            cluster_model = self.SURF_Cluster
            classifier = self.SVM_SURF
        elif model == "ORB":
            cluster_model = self.ORB_Cluster
            classifier = self.SVM_ORB

        img_clustered_words = cluster_model.get_img_clustered_words([image_desc])
        X = cluster_model.get_img_bow_hist(img_clustered_words,1000)
        y_pred = classifier.clf.predict(X)
        return y_pred[0]

    def features_extraction(self, image_name, model, resize):

        gray_image = cv2.imread(image_name,0)
        if gray_image is not None:
            if model == "SIFT":
                mdl = cv2.xfeatures2d.SIFT_create()
            elif model == "SURF":
                mdl = cv2.xfeatures2d.SURF_create(extended = True, hessianThreshold = 400)
            elif model == "ORB":
                mdl = cv2.ORB_create(1000)
            if resize:
                gray_image = cv2.resize(gray_image, resize, interpolation=cv2.INTER_AREA)

            kp, desc = mdl.detectAndCompute(gray_image, None)

            if len(kp) > 0:
                return desc
        return "ERROR"

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = "Waste Sorting"
        self.setWindowTitle(self.title)
        self.initGui()
        self.show()
        self.CNN = CNN()
        self.SVM = SVM()
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
            self.updateBins(self.SVM.predict(self.imagePath,self.featuresExtractionComboBox.currentText()))

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
