from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sys import stdout

def features_extraction(model, fnames, data, labels, imgs_path, resize):
    print("The start of features extraction")
    for category in fnames:
        files = fnames[category]
        n = len(files)
        for i in range(n):
            progressBar(n,i+1,category)
            file_name = files[i]
            gray_image = cv2.imread(file_name,0)
            if gray_image is not None:
                """
                Create SIFT model and extract images features
                kp is the keypoints
                desc is the SIFT descriptors, they're 128-dimensional vectors
                """
                if model == "SIFT":
                    mdl = cv2.xfeatures2d.SIFT_create()
                elif model == "SURF":
                    mdl = cv2.xfeatures2d.SURF_create(extended = True, hessianThreshold = 400)
                elif model == "ORB":
                    mdl = cv2.ORB_create(1000)
                if resize:
                    """
                    En comparant des images de même taille, nous pouvons aussi faire
                    des images plus grandes ou plus petites pour plus de précision
                    ou plus de vitesse.
                    """
                    gray_image = cv2.resize(gray_image, resize, interpolation=cv2.INTER_AREA)

                kp, desc = mdl.detectAndCompute(gray_image, None)

                if len(kp) > 0:
                    data.append(desc)
                    labels.append(category)
                    imgs_path.append(file_name)
    print("The end of the extraction")

def progressBar(total, progress, category):

    barLength, status = 20, ""
    progress = progress / total

    if progress >= 1:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\rProcessing of {} items [{}] {:.0f}% {}".format(str(total)+ " " + category,"#" * block + "-" * (barLength - block), round(progress * 100, 0),status)
    stdout.write(text)
    stdout.flush()


class ImageClassifierModel(object):

    def __init__(self):

        self.clf = None
        self.accuracy = None

    def createAndfit(self,training_data,training_labels,C,kernel,Gamma):

        """
        Create and fit the SVM
        """

        self.clf = svm.SVC(C, kernel, gamma = Gamma)
        self.clf.fit(training_data,training_labels)

    def best_estimator(self,training_data,training_labels):
        #["linear", "poly", "rbf", "sigmoid"]
        param_grid = {'kernel' : ["rbf"],
                      'C' : [1, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma' : [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 1e-2, 5e-1]}
        self.clf = GridSearchCV(svm.SVC(),param_grid, n_jobs = -1)
        self.clf = self.clf.fit(training_data,training_labels)
        print(self.clf.best_estimator_)


    def predictAndShow(self,X,y,imgs_path, resize):

        for x,category,img_path in zip(X,y,imgs_path):
            class_prediction = self.clf.predict([x])
            self.show(img_path,class_prediction,category,resize)

    def predict(self,X,y):
        y_pred = self.clf.predict(X)
        self.accuracy = round(self.clf.score(X, y)*100,2)
        return y_pred


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

    def show_confusion_matrix(self,y,y_pred):

        class_labels = self.clf.classes_.tolist()
        n = len(class_labels)
        confusionMatrix = confusion_matrix(y,y_pred,class_labels)
        fig, ax = plt.subplots()
        im = ax.imshow(confusionMatrix)

        # We want to show all ticks...
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        # ... and label them with the respective list entries
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, confusionMatrix[i, j],ha="center", va="center", color="w")

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

class ClusterModel(object):

    def __init__(self):

        self.cluster = None

    def createAndfit(self,training_data,n_clusters):

        all_train_descriptors = [desc for desc_list in training_data for desc in desc_list]
        all_train_descriptors = np.array(all_train_descriptors)

        self.cluster = KMeans(n_clusters,n_jobs=-1)
        self.cluster.fit(all_train_descriptors)

    def get_img_clustered_words(self,training_data):
        return [self.cluster.predict(raw_words) for raw_words in training_data]

    def get_img_bow_hist(self,img_clustered_words,n_clusters):
        return np.array([np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    def save_model(self,path):

        joblib.dump(self.cluster, path)

    def load_model(self,path):

        self.cluster = joblib.load(path)
