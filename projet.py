import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
import cv2
import csv
import os
from os import listdir
from os.path import isfile, join
from subprocess import check_output
import matplotlib.pyplot as plt


def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))


#plt.imshow(octo_front_gray, cmap='gray');


def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

#------------------------------------------------------------------------------

def to_gray(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()

    # kp is the keypoints
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

#Loading data

print(check_output(["ls", "waste-classification-data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
mydirvalues = [d for d in os.listdir(os.path.dirname(os.path.abspath(__file__)))]
#print(mydirvalues)
onlyfiles = [f for f in listdir("waste-classification-data/DATASET/TRAIN/") if isfile(join("waste-classification-data/DATASET/TRAIN/", f))]
#print(onlyfiles)

dir_names = [d for d in listdir("waste-classification-data/DATASET/TRAIN/") if not isfile(join("waste-classification-data/DATASET/TRAIN/", d))]
#print(dir_names)

file_paths = {}
class_num = 0
for d in dir_names:
     fnames = [f for f in listdir("waste-classification-data/DATASET/TRAIN/"+d+"/") if isfile(join("waste-classification-data/DATASET/TRAIN/"+d+"/", f))]
     #print(fnames)
     file_paths[(d, class_num, "waste-classification-data/DATASET/TRAIN/"+d+"/")] = fnames
     class_num += 1

#Feature Extraction

# General steps:
# Extract feature from each file as SIFT.
# map each to feature space... and train some kind of classifier on that. SVM is a good choice.
# do the same for each feature in test set...

training_data = np.array([])
training_labels = np.array([])

for key in file_paths:
    print(key)
    category = key[1]
    directory_path = key[2]
    file_list = file_paths[key]
    # shuffle this list, so we get random examples
    np.random.shuffle(file_list)

    # Stop early, while testing, so it doesn't take FOR-EV-ER (FOR-EV-ER)
    i = 0

    # read in the file and get its SIFT features
    for fname in file_list:
        fpath = directory_path + fname
        print(fpath)
        print("Category = " + str(category))
        # extract features!
        gray = to_gray(fpath)
        gray = cv2.resize(gray, (400, 250))  # resize so we're always comparing same-sized images
                                             # Could also make images larger/smaller
                                             # to tune for greater accuracy / more speedd

        detector = cv2.xfeatures2d.SIFT_create()
        kp, des = detector.detectAndCompute(gray, None)

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

        # early stop
        i += 1
        if i > 50:
            break

#Fit the SVM

# Alright! Now we've got features extracted and labels
X = training_data
y = training_labels
y = y.reshape(y.shape[0],)

# Create and fit the SVM
# Fitting should take a few minutes
clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
clf.fit(X,y)

#Make a Prediction

# Now, extract one of the images and predict it
gray = to_gray('waste-classification-data/DATASET/TEST/O/O_12583.jpg')
kp, des = detector.detectAndCompute(gray, None)
#show_sift_features(gray,cv2.imread(('waste-classification-data/DATASET/TEST/O/O_12583.jpg')),kp)
des = des[0:100, :]   # trim vector so all are same size
vector_data = des.reshape(1, 12800)

print("Linear SVM Prediction:")
print(clf.predict(vector_data))        # prints highest probability class, only
print(clf.predict_proba(vector_data))  # shows all probabilities for each class.
                                       #    need this for the competition


#Save the SVM for later Use

# save SVM model
# joblib.dump(clf, 'filename.pkl')
# to load SVM model, use:  clf = joblib.load('filename.pkl')

#Predict the whole Data Set

# early stoppage...
# only do 10
"""
i = 0
for f in fnames:
    file_name = "waste-classification-data/DATASET/TEST/" + f
    print("---Evaluating File at: " + file_name)
    gray = cv2.imread(file_name, 0)  # Correct is LAG --> Class 3
    gray = cv2.resize(gray, (400, 250))  # resize so we're always comparing same-sized images
    kp1, des1 = detector.detectAndCompute(gray, None)

    # ensure we have at least 100 keypoints to analyze
    if len(kp1) < 100:
        # and duplicate some points if necessary
        current_len = len(kp1)
        vectors_needed = 100 - current_len
        repeated_vectors = des1[0:vectors_needed, :]
        # concatenate repeats onto des1
        while len(des1) < 100:
            des1 = np.concatenate((des1, repeated_vectors), axis=0)
        # duplicate data just so we can run the model.
        des1[current_len:100, :] = des1[0:vectors_needed, :]

    np.random.shuffle(des1)  # shuffle the vector so we get a representative sample
    des1 = des1[0:100, :]   # trim vector so all are same size
    vector_data = des1.reshape(1, 12800)
    print("Linear SVM Prediction:")
    print(clf.predict(vector_data))
    svm_prediction = clf.predict_proba(vector_data)
    print(svm_prediction)

    # format list for csv output
    csv_output_list = []
    csv_output_list.append(f)
    for elem in svm_prediction:
        for value in elem:
            csv_output_list.append(value)

    # append filename to make sure we have right format to write to csv
    print("CSV Output List Formatted:")
    print(csv_output_list)

    # and append this file to the output_list (of lists)
    prediction_output_list.append(csv_output_list)

    # Uncomment to stop early
    if i > 10:
        break
    i += 1
"""
