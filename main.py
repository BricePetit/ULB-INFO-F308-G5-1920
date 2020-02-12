import constants
from Parser import *
from Model import *
from methods import *
import sys

#------------------------------------------------------------------------------

p = Parser(constants.PATH)
class_names = p.getClassNames()
accuracy = 0
max_iter = 5

for it in range(max_iter):
    print("Iteration :",it+1)
    p.dataset_split(0.2)
    train_file_paths = p.getTrainingFiles()
    test_fnames = p.getTestingFiles()

    training_data, training_labels = [],[]
    #features_extraction(train_file_paths, training_data, training_labels)

    print("Cluster model processing")
    cluster_model = ClusterModel(constants.N_CLUSTERS)
    #cluster_model.createAndfit(training_data)
    #cluster_model.save_model('cluster-model/cluster.pkl')
    cluster_model.load_model('cluster-model/cluster.pkl')
    #img_clustered_words = cluster_model.get_img_clustered_words(training_data)
    #X_train = cluster_model.get_img_bow_hist(img_clustered_words)
    #y_train = np.array(training_labels).transpose()

    print("SVM model processing")
    m = ImageClassifierModel()
    m.load_model('classification-model/classification.pkl')
    #m.createAndfit(X_train,y_train)
    clf = m.getClassifier()
    #m.save_model('classification-model/classification.pkl')

    testing_data, testing_labels =  [],[]
    #test_fnames = [(sys.argv[1],sys.argv[2])]
    features_extraction(test_fnames, testing_data, testing_labels)
    img_clustered_words = cluster_model.get_img_clustered_words(testing_data)
    X_test = cluster_model.get_img_bow_hist(img_clustered_words)
    y_test = np.array(testing_labels).transpose()
    imgs_path = [img for category in test_fnames for img in test_fnames[category]]
    accuracy += predict(clf,X_test,y_test,class_names,it,imgs_path)

average_accuracy = round(accuracy / max_iter * 100,2)
print("Average accuracy: {}%".format(average_accuracy))
