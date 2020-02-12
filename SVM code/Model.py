from sklearn import svm
from sklearn.cluster import KMeans
import joblib
import numpy as np

class ImageClassifierModel(object):

    def __init__(self):

        self.clf = None

    def createAndfit(self,training_data,training_labels):

        """
        Create and fit the SVM
        """

        kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]

        self.clf = svm.SVC(kernel= kernels[0], C = 1.0, probability=True)
        self.clf.fit(training_data,training_labels)

    def getClassifier(self):

        return self.clf

    def save_model(self,path):

        """
        Save the SVM for later Use
        """

        joblib.dump(self.clf, path)

    def load_model(self,path):

        """
        Load the SVM model
        """

        self.clf = joblib.load(path)

class ClusterModel(object):

    def __init__(self, n_clusters):

        self.n_clusters = n_clusters
        self.cluster = None

    def createAndfit(self,training_data):

        all_train_descriptors = [desc for desc_list in training_data for desc in desc_list]
        all_train_descriptors = np.array(all_train_descriptors)

        self.cluster = KMeans(self.n_clusters)
        self.cluster.fit(all_train_descriptors)

    def get_img_clustered_words(self,training_data):
        return [self.cluster.predict(raw_words) for raw_words in training_data]

    def get_img_bow_hist(self,img_clustered_words):
        return np.array([np.bincount(clustered_words, minlength=self.n_clusters) for clustered_words in img_clustered_words])

    def save_model(self,path):

        joblib.dump(self.cluster, path)

    def load_model(self,path):

        self.cluster = joblib.load(path)
