import joblib
from Model import *
from Parser import *
from sklearn.model_selection import train_test_split
# ------------------------------------------------------------------------------
# DIMENSIONS = (HEIGHT, WIDTH)
DIMENSIONS = (384, 512)
MAX_ITER = 1
C, KERNEL, GAMMA, k = 1000, "rbf", 1e-6, 1000
accuracy = []
cluster_models = ["1000-SIFT.pkl", "1000-SURF.pkl", "1000-ORB.pkl"]
models = ["SIFT", "SURF", "ORB"]
features_models = ["SIFT_FEATURES.pkl", "SURF_FEATURES.pkl", "ORB_FEATURES.pkl"]
model = models[0]
# ------------------------------------------------------------------------------
classifier = ImageClassifierModel(model, k)
classifier.load()

p = Parser("../dataset/")

# dataset, labels, files_path = [], [], []
# features_extraction(model, p.files, dataset, labels, files_path, DIMENSIONS)

train = joblib.load("SIFT_FEATURES.pkl")
dataset, labels, files_path = train[0], train[1], train[2]

print("Image dimensions : {} - k : {}".format(DIMENSIONS, k))

for it in range(MAX_ITER):
    print("Iteration :", it + 1)
    training_data, testing_data, training_labels, testing_labels, training_imgPath, testing_imgPath = \
        train_test_split(dataset, labels, files_path, test_size=0.2)

    print("SVM model processing ...")
    print("C : {} - Kernel : {} - Gamma : {}".format(C, KERNEL, GAMMA))

    classifier.create_and_fit_svm(training_data, training_labels, C, KERNEL, GAMMA)
    # classifier.predictAndShow(X_test,y_test,testing_imgPath,DIMENSIONS)
    y_pred = classifier.predict_all(testing_data, testing_labels)
    classifier.show_confusion_matrix(testing_labels, y_pred)
    accuracy.append(classifier.accuracy)

average_accuracy = round(sum(accuracy) / MAX_ITER, 2)
print("Average accuracy: {}%".format(average_accuracy))
