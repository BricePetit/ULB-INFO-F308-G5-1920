import numpy as np # linear algebra
from sklearn import svm
import cv2
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import joblib


def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(cv2.imread(img), cv2.CV_32S))

def show_gray_img(img):
    """Convenience function to display a typical gray image"""
    return plt.imshow(cv2.cvtColor(cv2.imread(img), cmap='gray'))

def to_gray(img_path,size = False):
    """Convert the image to greyscale and resize it"""
    if not size:
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.resize(cv2.imread(img_path), size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gen_sift_features(gray_img):
    """Create SIFT model and extract images features
       kp is the keypoints
       desc is the SIFT descriptors, they're 128-dimensional vectors
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def duplicate_kp(kp,des):
    """Duplicate some image keypoints"""
    current_len = len(kp)
    vectors_needed = 100 - current_len
    repeated_vectors = des[0:vectors_needed, :]

    while len(des) < 100:
        des = np.concatenate((des, repeated_vectors), axis=0)
    des[current_len:100, :] = des[0:vectors_needed, :]

    return kp,des

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

def createAndfit(training_data,training_labels):
    """Create and fit the SVM"""
    clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
    clf.fit(training_data,training_labels.reshape(training_labels.shape[0],))
    return clf

def save_model(clf,model_path):
    """Save the SVM for later Use"""
    joblib.dump(clf, model_path)

def load_model(model_path):
    """Load the SVM model"""
    return joblib.load(model_path)

def show(file_name,gray_img,kp,class_prediction,svm_prediction,file_class):
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    imgplot = show_rgb_img(file_name)
    a.set_title('Image of a {0}'.format(file_class))
    plt.axis('off')
    a = fig.add_subplot(1, 3, 2)
    imgplot = show_sift_features(gray_img,cv2.imread(file_name),kp)
    a.set_title('Sift features')
    plt.axis('off')
    x,y = a.axis()[1] + 30 , a.axis()[2]/2 + 10
    predictions = "\n".join(str(round(elem,2)) for elem in svm_prediction[0])

    txt = "Linear SVM Prediction: \nHighest probability class: \n{0} \nAll probabilities for each class: \n{1} ".format(class_prediction,predictions)
    plt.text(x, y, txt, dict(size=10), bbox=dict(facecolor='red', alpha=0.5))
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def show_matrix_prediction(class_name,score_matrix,score):

    fig, ax = plt.subplots()
    im = ax.imshow(score_matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_name)))
    ax.set_yticks(np.arange(len(class_name)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(class_name)
    ax.set_yticklabels(class_name)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_name)):
        for j in range(len(class_name)):
            text = ax.text(j, i, score_matrix[i, j],ha="center", va="center", color="w")


    ax.set_title("Prediction Matrix | Prediction score: {0}%".format(score/np.sum(score_matrix)*100))
    fig.tight_layout()
    plt.show()

def create_dataset(dataset):
    files = [f for f in listdir(dataset) if isfile(join(dataset, f))]
    dir_names = [d for d in listdir(dataset) if not isfile(join(dataset, d))]

    class_name = [d for d in dir_names]
    class_num = 0

    train_file_paths = {}
    test_fnames = []
    size = 2/3

    for d in dir_names:
        fnames = [f for f in listdir(dataset+d+"/") if isfile(join(dataset+d+"/", f))]
        #names = [join(dataset+d+"/", f) for f in listdir(dataset+d+"/") if isfile(join(dataset+d+"/", f))]
        #check_files(names)
        nb = int(len(fnames) * size)
        train_file_paths[(d, class_num, dataset+d+"/")] = fnames[:nb]
        class_num += 1
        test_fnames += [(d,join(dataset+d+"/", f)) for f in fnames[nb:]]

    return class_name,train_file_paths,test_fnames

def create_dataset_2():
    train_dataset = "waste-classification-data/DATASET/TRAIN/"
    test_dataset =  "waste-classification-data/DATASET/TEST/"

    train_files = [f for f in listdir(train_dataset) if isfile(join(train_dataset, f))]
    test_files = [f for f in listdir(test_dataset) if isfile(join(test_dataset, f))]

    train_dir_names = [d for d in listdir(train_dataset) if not isfile(join(train_dataset, d))]
    test_dir_names = [d for d in listdir(test_dataset) if not isfile(join(test_dataset, d))]

    train_file_paths = {}

    class_name = [d for d in train_dir_names]
    class_num = 0
    for d in train_dir_names:
         train_fnames = [f for f in listdir(train_dataset+d+"/") if isfile(join(train_dataset+d+"/", f))]
         train_file_paths[(d, class_num, train_dataset+d+"/")] = train_fnames
         class_num += 1

    test_fnames = []
    for d in test_dir_names:
         test_fnames += [(d,join(test_dataset+d+"/", f)) for f in listdir(test_dataset+d+"/") if isfile(join(test_dataset+d+"/", f))]

    return class_name,train_file_paths,test_fnames

def check_files(lst):
    l = []
    for f in lst:
        img = cv2.imread(f)
        if img is None:
            l.append(f)
    print(l)
#------------------------------------------------------------------------------


#Loading data
class_name,train_file_paths,test_fnames = create_dataset("dataset-original/")
# dataset link: https://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE

#class_name,train_file_paths,test_fnames = create_dataset_2()

n = len(class_name)
score_matrix = np.zeros((n,n), dtype=float)
size = (400, 250)

#Feature extraction and training

training_data = np.array([])
training_labels = np.array([])

for key in train_file_paths:
    category, directory_path = key[1], key[2]
    file_list = train_file_paths[key]
    i = 0
    while i < 50:#len(file_list):
        fname = file_list[i]
        fpath = directory_path + fname
        #print(fpath)
        #print("Category = " + str(category))

        gray = to_gray(fpath,size)
        #comparing same-sized images, could also make
        # images larger/smaller to tune for greater accuracy / more speeded

        kp, des = gen_sift_features(gray)

        if len(kp) == 0:
            i += 1
            continue
        elif len(kp) < 100:
            # This is to make sure we have at least 100 keypoints to analyze
            kp,des = duplicate_kp(kp,des)

        des = des.astype(np.float64)
        # transform the data to float and shuffle all keypoints
        # so we get a random sampling from each image
        np.random.shuffle(des)
        des = des[0:100,:]
        # trim vector so all are same size
        vector_data = des.reshape(1,12800)
        list_data = vector_data.tolist()

        # We need to concatenate on the full list of features extracted from each image
        if len(training_data) == 0:
            training_data = np.append(training_data, vector_data)
            training_data = training_data.reshape(1,12800)
        else:
            training_data   = np.concatenate((training_data, vector_data), axis=0)

        training_labels = np.append(training_labels,category)
        i += 1

clf = createAndfit(training_data,training_labels)
#save_model(clf,'classification-model/classification.pkl')
#clf = load_model('classification-model/classification.pkl')

#Predict the whole Data Set
i = 0
score = 0
while i < 10:#len(test_fnames):
    file_name, file_class = test_fnames[i][1], test_fnames[i][0]
    gray = to_gray(file_name,size)
    #resize so we're always comparing same-sized images
    kp, des = gen_sift_features(gray)

    if len(kp) == 0:
        i += 1
        continue
    elif len(kp) < 100:
        # ensure we have at least 100 keypoints to analyze
        kp, des = duplicate_kp(kp,des)

    np.random.shuffle(des)
    # shuffle the vector so we get a representative sample
    des = des[0:100, :]
    # trim vector so all are same size
    vector_data = des.reshape(1, 12800)

    class_prediction = class_name[int(clf.predict(vector_data))]
    #highest probability class, only
    svm_prediction = clf.predict_proba(vector_data)
    # shows all probabilities for each class.

    score = score + 1 if class_prediction == file_class else score
    category = class_name.index(file_class)
    predict_category = class_name.index(class_prediction)
    score_matrix[category][predict_category] += 1
    #show(file_name,gray,kp,class_prediction,svm_prediction,file_class)
    i += 1

print(score,len(test_fnames))
#show_matrix_prediction(class_name,score_matrix,score)
