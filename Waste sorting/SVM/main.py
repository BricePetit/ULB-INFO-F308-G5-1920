import joblib
from Model import *
from Parser import *
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------

TRAINING_PATH = "../dataset/"
# DIMENSIONS = (HEIGHT, WIDTH)
DIMENSIONS = (384, 512)
MAX_ITER = 1
C = 1000
KERNEL = "rbf"
GAMMA = 1e-6
k = 1000
accuracy = []
cluster_models = ["1000-SIFT.pkl", "1000-SURF.pkl", "1000-ORB.pkl"]
models = ["SIFT", "SURF", "ORB"]
features_models = ["SIFT_FEATURES.pkl", "SURF_FEATURES.pkl", "ORB_FEATURES.pkl"]
model = models[2]
cluster = cluster_models[2]
features = features_models[2]
# ------------------------------------------------------------------------------
p = Parser(TRAINING_PATH)
file_paths = p.files
# dataset, labels, imgs_path = [],[],[]
# features_extraction(model, file_paths, dataset, labels, imgs_path, resize = DIMENSIONS)

# joblib.dump([dataset, labels, imgs_path], features)
train = joblib.load(features)
# dataset, labels, imgs_path = train[0], train[1], train[2]
data, labels = train[0], train[1]

print("Cluster model processing ...")

print("Image dimensions : {} - k : {}".format(DIMENSIONS, k))
cluster_model = ClusterModel()
# cluster_model.createAndfit(dataset,k)
# cluster_model.save_model('cluster-model/{}-{}.pkl'.format(k,model))
cluster_model.load_model('cluster-model/' + cluster)

for it in range(MAX_ITER):
    print("Iteration :", it + 1)
    # training_data, testing_data, training_labels, testing_labels, training_imgPath, testing_imgPath = \
    # train_test_split(dataset, labels, imgs_path, test_size = 0.2)
    training_data, testing_data, training_labels, testing_labels, = \
        train_test_split(data, labels, test_size=0.2)
    training_imgPath, testing_imgPath = [], []

    img_clustered_words = cluster_model.get_img_clustered_words(training_data)
    X_train = cluster_model.get_img_bow_hist(img_clustered_words, k)
    y_train = np.array(training_labels).transpose()

    img_clustered_words = cluster_model.get_img_clustered_words(testing_data)
    X_test = cluster_model.get_img_bow_hist(img_clustered_words, k)
    y_test = np.array(testing_labels).transpose()

    print("SVM model processing ...")
    print("C : {} - Kernel : {} - Gamma : {}".format(C, KERNEL, GAMMA))
    classifier = ImageClassifierModel()
    classifier.create_and_fit(X_train, y_train, C, KERNEL, GAMMA)
    # classifier.best_estimator(X_train,y_train)
    # classifier.save_model('classification-model/{}-SVM-{}.pkl'.format(k, model))
    # classifier.load_model('classification-model/1000-classification.pkl')
    # classifier.predictAndShow(X_test,y_test,testing_imgPath,DIMENSIONS)
    y_pred = classifier.predict(X_test, y_test)
    classifier.show_confusion_matrix(y_test, y_pred)
    accuracy.append(classifier.accuracy)

average_accuracy = round(sum(accuracy) / MAX_ITER, 2)
print("Average accuracy: {}%".format(average_accuracy))
