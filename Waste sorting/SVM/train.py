from Model import *
from Parser import *
from sklearn.model_selection import train_test_split
# ------------------------------------------------------------------------------
# DIMENSIONS = (HEIGHT, WIDTH)
DIMENSIONS = (384, 512)
C, KERNEL, GAMMA, k = 1000, "rbf", 1e-6, 1000
models = ["SIFT", "SURF", "ORB"]
model = models[0]
# ------------------------------------------------------------------------------
classifier = ImageClassifierModel(model, k)
p = Parser("../dataset/")
file_paths = p.files

dataset, labels, files_path = [], [], []
features_extraction(model, file_paths, dataset, labels, files_path, resize=DIMENSIONS)

print("Cluster model processing ...")

print("Image dimensions : {} - k : {}".format(DIMENSIONS, k))
classifier.create_and_fit_cluster(dataset)
classifier.save_cluster()

training_data, testing_data, training_labels, testing_labels, training_imgPath, testing_imgPath = \
        train_test_split(dataset, labels, files_path, test_size=0.2)

print("SVM model processing ...")
print("C : {} - Kernel : {} - Gamma : {}".format(C, KERNEL, GAMMA))
classifier.create_and_fit(training_data, testing_data, C, KERNEL, GAMMA)
# classifier.best_estimator(training_data, testing_data)
classifier.save_svm()
