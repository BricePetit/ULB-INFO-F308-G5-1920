from sklearn import svm
import joblib

class ImageClassifierModel(object):

    def __init__(self):

        self.clf = None

    def createAndfit(self,training_data,training_labels):

        """
        Create and fit the SVM
        """

        kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]

        self.clf = svm.SVC(kernel= kernels[0], C = 1.0, probability=True)
        self.clf.fit(training_data,training_labels.reshape(training_labels.shape[0],))

    def getClassifier(self):

        return self.clf

    def save_model(self,clf,path):

        """
        Save the SVM for later Use
        """

        joblib.dump(clf, path)

    def load_model(self,path):

        """
        Load the SVM model
        """

        self.clf = joblib.load(path)
