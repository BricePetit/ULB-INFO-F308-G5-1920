from sys import stdout

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def extract(image_name, model, resize):
    gray_image = cv2.imread(image_name, 0)
    if gray_image is not None:
        if model == "SIFT":
            mdl = cv2.xfeatures2d.SIFT_create()
        elif model == "SURF":
            mdl = cv2.xfeatures2d.SURF_create(extended=True, hessianThreshold=400)
        elif model == "ORB":
            mdl = cv2.ORB_create(1000)
        if resize:
            gray_image = cv2.resize(gray_image, resize, interpolation=cv2.INTER_AREA)

        kp, desc = mdl.detectAndCompute(gray_image, None)
        return kp, desc
    raise


def features_extraction(model, files, data, labels, files_path, resize):
    print("The start of features extraction")
    for category in files:
        category_files = files[category]
        n = len(category_files)
        for i in range(n):
            progress_bar(n, i + 1, category)
            file_name = category_files[i]
            kp, desc = extract(file_name, model, resize)
            if len(kp) > 0:
                data.append(desc)
                labels.append(category)
                files_path.append(file_name)
    print("The end of the extraction")


def progress_bar(total, progress, category):
    bar_length, status = 20, ""
    progress = progress / total

    if progress >= 1:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    text = "\rProcessing of {} items [{}] {:.0f}% {}".format(str(total) + " " + category,
                                                             "#" * block + "-" * (bar_length - block),
                                                             round(progress * 100, 0), status)
    stdout.write(text)
    stdout.flush()


class ImageClassifierModel(object):

    def __init__(self, model=None, k=None):

        self.n_clusters = k
        self.model = model
        self.cluster = None
        self.clf = None
        self.accuracy = None

    def save_cluster(self):

        """
        Save the K-Means for later Use
        """

        joblib.dump(self.cluster, "SVM/cluster-model/{}-{}.pkl".format(self.n_clusters, self.model))

    def save_svm(self):

        """
        Save the SVM for later Use
        """

        joblib.dump(self.clf, "SVM/classification-model/{}-SVM-{}.pkl".format(self.n_clusters, self.model))

    def load(self):

        """
        Load the K-Means and SVM model
        """

        self.cluster = joblib.load("SVM/cluster-model/{}-{}.pkl".format(self.n_clusters, self.model))
        self.clf = joblib.load("SVM/classification-model/{}-SVM-{}.pkl".format(self.n_clusters, self.model))

    def create_and_fit_cluster(self, training_data):

        all_train_descriptors = np.array([desc for desc_list in training_data for desc in desc_list])

        self.cluster = KMeans(self.n_clusters, n_jobs=-1)
        self.cluster.fit(all_train_descriptors)

    def get_img_bow_hist(self, training_data):

        img_clustered_words = [self.cluster.predict(raw_words) for raw_words in training_data]
        return np.array([np.bincount(clustered_words, minlength=self.n_clusters) for clustered_words in img_clustered_words])

    def create_and_fit_svm(self, training_data, training_labels, c, kernel, gamma):

        """
        Create and fit the SVM
        """
        X_train = self.get_img_bow_hist(training_data)
        y_train = np.array(training_labels).transpose()

        self.clf = svm.SVC(c, kernel, gamma=gamma)
        self.clf.fit(X_train, y_train)

    def predict(self, img_path):

        image_desc = extract(img_path, self.model, resize=(384, 512))[1]
        X = self.get_img_bow_hist([image_desc])
        y_pred = self.clf.predict(X)
        return y_pred[0]

    def best_estimator(self, training_data, training_labels):
        # ["linear", "poly", "rbf", "sigmoid"]
        param_grid = {'kernel': ["rbf"],
                      'C': [1, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 1e-2, 5e-1]}

        X_train = self.get_img_bow_hist(training_data)
        y_train = np.array(training_labels).transpose()
        self.clf = GridSearchCV(svm.SVC(), param_grid, n_jobs=-1)
        self.clf = self.clf.fit(X_train, y_train)
        print(self.clf.best_estimator_)

    def predict_and_show(self, X, y, files_path, resize):

        for x, category, img_path in zip(X, y, files_path):
            class_prediction = self.clf.predict([x])
            self.show(img_path, class_prediction, category, resize)

    def predict_all(self, training_data, training_labels):

        X = self.get_img_bow_hist(training_data)
        y = np.array(training_labels).transpose()

        y_pred = self.clf.predict(X)
        self.accuracy = round(self.clf.score(X, y) * 100, 2)
        return y_pred

    def show(self, file_name, class_prediction, category, size):

        image = cv2.resize(cv2.imread(file_name), size)
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.CV_32S))
        a.set_title('{0} class'.format(category))
        plt.axis('off')

        x, y = a.axis()[1] + 30, a.axis()[2] / 2 + 10
        txt = "SVM Prediction : {} class".format(class_prediction[0])

        plt.text(x, y, txt, dict(size=10), bbox=dict(facecolor='red', alpha=0.5))
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    def show_confusion_matrix(self, y, y_pred):

        class_labels = self.clf.classes_.tolist()
        n = len(class_labels)
        matrix = confusion_matrix(y, y_pred, class_labels)
        fig, ax = plt.subplots()
        ax.imshow(matrix)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(n):
            for j in range(n):
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="w")

        ax.set_title("Prediction accuracy: {0}%".format(self.accuracy))
        fig.tight_layout()
        # fig.savefig("Figures/fig"+str(it)+".png")
        plt.show()
