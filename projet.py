import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
import cv2
import csv
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(cv2.imread(img), cv2.CV_32S))

def show_gray_img(img):
    """Convenience function to display a typical gray image"""
    return plt.imshow(cv2.cvtColor(cv2.imread(img), cmap='gray'))

def to_gray(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

def gen_sift_features(gray_img):

    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

#------------------------------------------------------------------------------


#Loading data

#print(check_output(["ls", "waste-classification-data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#mydirvalues = [d for d in os.listdir(os.path.dirname(os.path.abspath(__file__)))]
#print(mydirvalues)

train_dataset = "waste-classification-data/DATASET/TRAIN/"
test_dataset =  "waste-classification-data/DATASET/TEST/"

train_files = [f for f in listdir(train_dataset) if isfile(join(train_dataset, f))]
test_files = [f for f in listdir(test_dataset) if isfile(join(test_dataset, f))]

train_dir_names = [d for d in listdir(train_dataset) if not isfile(join(train_dataset, d))]
test_dir_names = [d for d in listdir(test_dataset) if not isfile(join(test_dataset, d))]

train_file_paths = {}

class_num = 0
for d in train_dir_names:
     train_fnames = [f for f in listdir(train_dataset+d+"/") if isfile(join(train_dataset+d+"/", f))]
     train_file_paths[(d, class_num, train_dataset+d+"/")] = train_fnames
     class_num += 1

test_fnames = []
for d in test_dir_names:
     test_fnames += [join(test_dataset+d+"/", f) for f in listdir(test_dataset+d+"/") if isfile(join(test_dataset+d+"/", f))]

#Feature Extraction

training_data = np.array([])
training_labels = np.array([])

for key in train_file_paths:
    category, directory_path = key[1], key[2]
    file_list = train_file_paths[key]

    nb = 300 #len(file_list)
    for fname in file_list[:nb]:
        # read in the file and get its SIFT features
        np.random.shuffle(file_list)
        fpath = directory_path + fname
        #print(fpath)
        #print("Category = " + str(category))

        # extract features
        gray = to_gray(fpath)
        gray = cv2.resize(gray, (400, 250))
        # resize so we're always comparing same-sized images, could also make
        # images larger/smaller to tune for greater accuracy / more speedd

        kp, des = gen_sift_features(gray)

        # This is to make sure we have at least 100 keypoints to analyze
        # could also duplicate a few features if needed to hit a higher value
        if len(kp) < 100:
            continue

        # transform the data to float and shuffle all keypoints
        # so we get a random sampling from each image
        des = des.astype(np.float64)
        np.random.shuffle(des)
        des = des[0:100,:] # trim vector so all are same size
        vector_data = des.reshape(1,12800)
        list_data = vector_data.tolist()

        # We need to concatenate ont the full list of features extracted from each image
        if len(training_data) == 0:
            training_data = np.append(training_data, vector_data)
            training_data = training_data.reshape(1,12800)
        else:
            training_data   = np.concatenate((training_data, vector_data), axis=0)

        training_labels = np.append(training_labels,category)

# Create and fit the SVM

clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
clf.fit(training_data,training_labels.reshape(training_labels.shape[0],))

#Predict the whole Data Set

nb = 100 #len(test_fnames)
for file_name in test_fnames[:nb]:
    np.random.shuffle(test_fnames)
    print("---Evaluating File at: " + file_name)
    gray = to_gray(file_name)
    gray = cv2.resize(gray, (400, 250))  # resize so we're always comparing same-sized images
    kp, des = gen_sift_features(gray)

    # ensure we have at least 100 keypoints to analyze
    if len(kp) < 100:
        # and duplicate some points if necessary
        current_len = len(kp)
        vectors_needed = 100 - current_len
        repeated_vectors = des[0:vectors_needed, :]
        # concatenate repeats onto des
        while len(des) < 100:
            des = np.concatenate((des, repeated_vectors), axis=0)
        # duplicate data just so we can run the model.
        des[current_len:100, :] = des[0:vectors_needed, :]

    np.random.shuffle(des)  # shuffle the vector so we get a representative sample
    des = des[0:100, :]   # trim vector so all are same size
    vector_data = des.reshape(1, 12800)

    #show_rgb_img(file_name)
    #plt.show()
    print("Linear SVM Prediction:")
    print(clf.predict(vector_data)) # prints highest probability class, only
    svm_prediction = clf.predict_proba(vector_data) # shows all probabilities for each class.
    print(svm_prediction)

    """
    # format list for csv output
    csv_output_list = []
    csv_output_list.append(file_name)
    for elem in svm_prediction:
        for value in elem:
            csv_output_list.append(value)

    # append filename to make sure we have right format to write to csv
    print("CSV Output List Formatted:")
    print(csv_output_list)

    # and append this file to the output_list (of lists)
    prediction_output_list.append(csv_output_list)
    """

#Save the SVM for later Use

# save SVM model
# joblib.dump(clf, 'filename.pkl')
# to load SVM model, use:  clf = joblib.load('filename.pkl')
