import constants
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sys import stdout

def gen_sift_features(image,resize = False):
    """
    Create SIFT model and extract images features
    kp is the keypoints
    desc is the SIFT descriptors, they're 128-dimensional vectors
    """
    sift = cv2.xfeatures2d.SIFT_create()

    if not resize:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.resize(image, constants.SIZE)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, desc = sift.detectAndCompute(gray_image, None)
    return kp, desc

def features_extraction(train_fnames, training_data, training_labels):

    for category in train_fnames:
        files = train_fnames[category]
        n = 100#len(files)
        for i in range(n):
            progressBar(n,i+1,category)
            file_name = files[i]
            img = cv2.imread(file_name)
            if img is not None:
                kp, des = gen_sift_features(img,resize = True)
                """
                En comparant des images de même taille, nous pouvons aussi faire
                des images plus grandes ou plus petites pour plus de précision
                ou plus de vitesse.
                """
                if len(kp) > 0:
                    training_data.append(des)
                    training_labels.append(category)

def predict(clf,X,y,class_names):

    confusion_matrix = np.zeros((len(class_names),len(class_names)), dtype=int)
    for i in range(len(X)):
        class_prediction = clf.predict([X[i]])
        """
        La probabilité de la classe la plus élevée seulement.
        """
        svm_prediction = clf.predict_proba([X[i]])
        """
        Les probabilités de chaque classe.
        """
        category = y[i]
        category_index = class_names.index(category)
        predict_category_index = class_names.index(class_prediction)
        confusion_matrix[category_index][predict_category_index] += 1
    #show(img,gray,kp,class_prediction,svm_prediction,category)
    accuracy = clf.score(X, y)
    #show_confusion_matrix(class_names,confusion_matrix,accuracy)
    return accuracy

def duplicate_kp(kp,des,min_kp):
    """
    Duplicate some image keypoints
    """
    current_len = len(kp)
    vectors_needed = min_kp - current_len
    repeated_vectors = des[0:vectors_needed, :]

    while len(des) < min_kp:
        des = np.concatenate((des, repeated_vectors), axis=0)
    des[current_len:min_kp, :] = des[0:vectors_needed, :]

    return kp,des

def show_rgb_img(img):
    """
    Convenience function to display a typical color image
    """
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

def show_gray_image(img):
    """
    Convenience function to display a typical gray image
    """
    return plt.imshow(cv2.cvtColor(img, cmap='gray'))

def show_sift_features(gray_image, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_image, kp, color_img.copy()))

def show(image,gray_image,kp,class_prediction,svm_prediction,file_class):

    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    imgplot = show_rgb_img(cv2.resize(image, size))
    a.set_title('Image of a {0}'.format(file_class))
    plt.axis('off')

    a = fig.add_subplot(1, 3, 2)
    imgplot = show_sift_features(gray_image,image,kp)
    a.set_title('Sift features')
    plt.axis('off')

    x,y = a.axis()[1] + 30 , a.axis()[2]/2 + 10
    predictions = "\n".join(str(round(elem,2)) for elem in svm_prediction[0])

    txt1 = "Linear SVM Prediction:\n"
    txt2 = "Highest probability class:\n{0}\n".format(class_prediction)
    txt3 = "All probabilities for each class:\n{0}".format(predictions)
    txt = txt1 + txt2 + txt3
    plt.text(x, y, txt, dict(size=10), bbox=dict(facecolor='red', alpha=0.5))
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def show_confusion_matrix(class_name,confusion_matrix,accuracy):

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

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
            text = ax.text(j, i, confusion_matrix[i, j],ha="center", va="center", color="w")

    ax.set_title("Prediction accuracy: {0}%".format(round(accuracy*100,2)))
    fig.tight_layout()
    plt.show()

def progressBar(total, progress, category):

    barLength, status = 20, ""
    progress = progress / total

    if progress >= 1:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\rProcessing of {} items [{}] {:.0f}% {}".format(str(total)+ " " + category,"#" * block + "-" * (barLength - block), round(progress * 100, 0),status)
    stdout.write(text)
    stdout.flush()
