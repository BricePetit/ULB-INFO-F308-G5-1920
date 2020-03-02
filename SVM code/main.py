from Parser import *
from Model import *
import sys
import joblib
#------------------------------------------------------------------------------
#"dataset-original/"
# dataset link: https://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE

TRAINING_PATH = "../data/"
TESTING_PATH = "../Test-data/"
SIZE = (384, 512)
MAX_ITER = 1
C = 1000
KERNEL = "rbf"  #kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
GAMMA = 1e-06
k = 1000
accuracy = []
#------------------------------------------------------------------------------
p = Parser(TRAINING_PATH)
p2 = Parser(TESTING_PATH)

train_file_paths = p.files
test_file_paths = p2.files

training_data, training_labels = [],[]
features_extraction(train_file_paths, training_data, training_labels, resize = SIZE)
testing_data, testing_labels =  [],[]
features_extraction(test_file_paths, testing_data, testing_labels, resize = SIZE)

print("Cluster model processing ...")

print("Image size : {} - k : {}".format(SIZE,k))
cluster_model = ClusterModel()
#cluster_model.createAndfit(training_data,k)
#cluster_model.save_model('cluster-model/{}-cluster.pkl'.format(k))
cluster_model.load_model('cluster-model/1000-cluster.pkl')

img_clustered_words = cluster_model.get_img_clustered_words(training_data)
X_train = cluster_model.get_img_bow_hist(img_clustered_words,k)
y_train = np.array(training_labels).transpose()

img_clustered_words = cluster_model.get_img_clustered_words(testing_data)
X_test = cluster_model.get_img_bow_hist(img_clustered_words,k)
y_test = np.array(testing_labels).transpose()

imgs_path = [img for category in test_file_paths for img in test_file_paths[category]]

for it in range(MAX_ITER):
    print("Iteration :",it+1)
    print("SVM model processing ...")
    print("C : {} - Kernel : {} - Gamma : {}".format(C,KERNEL,GAMMA))

    clf = ImageClassifierModel()
    clf.createAndfit(X_train,y_train,C,KERNEL,GAMMA)
    #clf.save_model('classification-model/{}-classification.pkl'.format(k))
    #clf.load_model('classification-model/1000-classification.pkl')
    y_pred = clf.predict(X_test,y_test)
    #y_pred = clf.predictAndShow(X_test,y_test,imgs_path, SIZE)
    clf.show_confusion_matrix(y_test,y_pred)
    accuracy.append(clf.accuracy)

average_accuracy = round(sum(accuracy) / MAX_ITER,2)
print("Average accuracy: {}%".format(average_accuracy))