import constants
from Parser import *
from training_test import *
from testing_test import *
from Model import *
import sys

#------------------------------------------------------------------------------

p = Parser(constants.PATH)
class_names = p.getClassNames()
accuracy = 0
max_iter = 2

for it in range(max_iter):
    print("Iteration :",it+1)
    p.dataset_split(0.2)
    train_file_paths = p.getTrainingFiles()
    test_fnames = p.getTestingFiles()

    all_descriptors, training_data, training_labels = train(train_file_paths)

    cluster_model = ClusterModel(constants.N_CLUSTERS)
    cluster_model.createAndfit(all_descriptors)
    #cluster_model.save_model('cluster-model/cluster.pkl')
    a = cluster_model.get_img_clustered_words(training_data)
    b = cluster_model.get_img_bow_hist(a)

    m = ImageClassifierModel()
    #m.load_model('classification-model/classification.pkl')
    #m.createAndfit(training_data,training_labels)
    m.createAndfit(b,training_labels)
    clf = m.getClassifier()
    #m.save_model('classification-model/classification.pkl')

    #test_fnames = [(sys.argv[1],sys.argv[2])]
    accuracy += predict(clf,test_fnames,class_names,cluster_model)

average_accuracy = round(accuracy / max_iter * 100,2)
print("Average accuracy: {}%".format(average_accuracy))
