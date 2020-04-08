from Parser import *
from Model import *
import sys
import joblib
from sklearn.model_selection import train_test_split
#------------------------------------------------------------------------------

TRAINING_PATH = "../data/"
#DIMENSIONS = (HEIGHT, WIDTH)
DIMENSIONS = (384, 512)
MAX_ITER = 1
C = 1000
KERNEL = "rbf"
GAMMA = 1e-6
k = 1000
accuracy = []
cluster_models = ["1000-SIFT.pkl","1000-SURF.pkl","1000-ORB.pkl"]
model = ["SIFT","SURF","ORB"]
#------------------------------------------------------------------------------
p = Parser(TRAINING_PATH)
file_paths = p.files
data, labels, imgs_path = [],[],[]
features_extraction(model[1],file_paths, data, labels, imgs_path, resize = DIMENSIONS)
print("Cluster model processing ...")

print("Image dimensions : {} - k : {}".format(DIMENSIONS,k))
cluster_model = ClusterModel()
#cluster_model.createAndfit(data,k)
#cluster_model.save_model('cluster-model/{}-{}.pkl'.format(k,model[1]))
cluster_model.load_model('cluster-model/'+ cluster_models[1])

for it in range(MAX_ITER):
    print("Iteration :",it+1)
    training_data, testing_data, training_labels, testing_labels, training_imgPath, testing_imgPath = \
    train_test_split(data, labels, imgs_path, test_size = 0.1)

    img_clustered_words = cluster_model.get_img_clustered_words(training_data)
    X_train = cluster_model.get_img_bow_hist(img_clustered_words,k)
    y_train = np.array(training_labels).transpose()

    img_clustered_words = cluster_model.get_img_clustered_words(testing_data)
    X_test = cluster_model.get_img_bow_hist(img_clustered_words,k)
    y_test = np.array(testing_labels).transpose()

    print("SVM model processing ...")
    print("C : {} - Kernel : {} - Gamma : {}".format(C,KERNEL,GAMMA))
    clf = ImageClassifierModel()
    clf.createAndfit(X_train,y_train,C,KERNEL,GAMMA)
    #clf.best_estimator(X_train,y_train)
    #clf.save_model('classification-model/{}-classification.pkl'.format(k))
    #clf.load_model('classification-model/1000-classification.pkl')
    #clf.predictAndShow(X_test,y_test,testing_imgPath,DIMENSIONS)
    y_pred = clf.predict(X_test,y_test)
    clf.show_confusion_matrix(y_test,y_pred)
    accuracy.append(clf.accuracy)

average_accuracy = round(sum(accuracy) / MAX_ITER,2)
print("Average accuracy: {}%".format(average_accuracy))
