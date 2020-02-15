from Parser import *
from Model import *
from methods import *
import sys
import joblib
from sklearn.model_selection import train_test_split
#------------------------------------------------------------------------------
#"dataset-original/"
# dataset link: https://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE

PATH = "../dataset-resized/"
SIZE = (384, 512)
MAX_ITER = 1
C = 1250
KERNEL = "rbf"  #kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
GAMMA =  1e-06
accuracy = []
#------------------------------------------------------------------------------
#p = Parser(sys.argv[1])
p = Parser(PATH)
class_names = p.getClassNames()

p.dataset_split(0.3)
train_file_paths = p.getTrainingFiles()
test_file_paths = p.getTestingFiles()
k = sum(len(train_file_paths[category]) for category in train_file_paths)
#test_file_paths = p.files

training_data, training_labels = [],[]
features_extraction(train_file_paths, training_data, training_labels, resize = False)

testing_data, testing_labels =  [],[]
features_extraction(test_file_paths, testing_data, testing_labels, resize = False)

print("Cluster model processing ...")

print("Image size : {} - k : {}".format(SIZE,k))
cluster_model = ClusterModel()
#cluster_model.createAndfit(training_data,k)
#cluster_model.save_model('cluster-model/{}-cluster.pkl'.format(k))
cluster_model.load_model('cluster-model/1766-cluster.pkl')

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
    m = ImageClassifierModel()
    m.createAndfit(X_train,y_train,C,KERNEL,GAMMA)
    m.save_model('classification-model/{}-classification.pkl'.format(k))
    #m.load_model('classification-model/classification.pkl')
    m.predict(X_test,y_test,imgs_path, SIZE)
    m.show_confusion_matrix()
    accuracy.append(m.accuracy)

average_accuracy = round(sum(accuracy) / MAX_ITER,2)
print("Average accuracy: {}%".format(average_accuracy))
