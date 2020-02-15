from sklearn import svm
from sklearn.cluster import KMeans
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ImageClassifierModel(object):

    def __init__(self):

        self.clf = None
        self.classes_labels = None
        self.confusion_matrix = None
        self.accuracy = None

    def fill(self):
        self.classes_labels = self.clf.classes_.tolist()
        n = len(self.classes_labels)
        self.confusion_matrix = np.zeros((n,n), dtype=int)

    def createAndfit(self,training_data,training_labels,C,kernel,Gamma):

        """
        Create and fit the SVM
        """
        self.clf = svm.SVC(C, kernel, gamma = Gamma)
        self.clf.fit(training_data,training_labels)
        self.fill()

    def predict(self,X,y,imgs_path, resize):

        for x,category,img_path in zip(X,y,imgs_path):
            class_prediction = self.clf.predict([x])
            """
            La probabilité de la classe la plus élevée seulement.
            """
            i = self.classes_labels.index(category)
            j = self.classes_labels.index(class_prediction)
            self.confusion_matrix[i][j] += 1
            #self.show(img_path,class_prediction,category,resize)
        self.accuracy = round(self.clf.score(X, y)*100,2)
        return self.accuracy

    def show(self,file_name,class_prediction,category,size):

        image = cv2.resize(cv2.imread(file_name),size)
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(cv2.cvtColor(image, cv2.CV_32S))
        a.set_title('{0} class'.format(category))
        plt.axis('off')

        x,y = a.axis()[1] + 30 , a.axis()[2]/2 + 10
        txt = "SVM Prediction : {} class".format(class_prediction[0])

        plt.text(x, y, txt, dict(size=10), bbox=dict(facecolor='red', alpha=0.5))
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    def show_confusion_matrix(self):

        n = len(self.classes_labels)
        fig, ax = plt.subplots()
        im = ax.imshow(self.confusion_matrix)

        # We want to show all ticks...
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        # ... and label them with the respective list entries
        ax.set_xticklabels(self.classes_labels)
        ax.set_yticklabels(self.classes_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, self.confusion_matrix[i, j],ha="center", va="center", color="w")

        ax.set_title("Prediction accuracy: {0}%".format(self.accuracy))
        fig.tight_layout()
        #fig.savefig("Figures/fig"+str(it)+".png")
        plt.show()

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
        self.fill()

class ClusterModel(object):

    def __init__(self):

        self.cluster = None

    def createAndfit(self,training_data,n_clusters):

        all_train_descriptors = [desc for desc_list in training_data for desc in desc_list]
        all_train_descriptors = np.array(all_train_descriptors)

        self.cluster = KMeans(n_clusters)
        self.cluster.fit(all_train_descriptors)

    def get_img_clustered_words(self,training_data):
        return [self.cluster.predict(raw_words) for raw_words in training_data]

    def get_img_bow_hist(self,img_clustered_words,n_clusters):
        return np.array([np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    def save_model(self,path):

        joblib.dump(self.cluster, path)

    def load_model(self,path):

        self.cluster = joblib.load(path)
