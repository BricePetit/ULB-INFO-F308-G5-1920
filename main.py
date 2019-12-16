
from Parser import *
from training import *
from testing import *
from Model import *
#------------------------------------------------------------------------------

p = Parser("dataset-original/")
class_names = p.getClassNames()
train_file_paths = p.getTrainingFiles()
test_fnames = p.getTestingFiles()
# dataset link: https://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE

training_data,training_labels = train(train_file_paths)

m = ImageClassifierModel()
#m.load_model('classification-model/classification.pkl')
#clf = m.createAndfit(training_data,training_labels)
clf = m.getClassifier()
#m.save_model(clf,'classification-model/classification.pkl')

predict(clf,test_fnames,class_names)
