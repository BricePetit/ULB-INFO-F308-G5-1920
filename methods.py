import numpy as np
import matplotlib.pyplot as plt
import cv2
from sys import stdout

size = (400, 250)

def gen_sift_features(gray_image):
    """Create SIFT model and extract images features
       kp is the keypoints
       desc is the SIFT descriptors, they're 128-dimensional vectors
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_image, None)
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

def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

def show_gray_image(img):
    """Convenience function to display a typical gray image"""
    return plt.imshow(cv2.cvtColor(img, cmap='gray'))

def to_gray(img,size = False):
    """Convert the image to greyscale and resize it"""
    if not size:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.resize(img, size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

def progressBar(total, progress, category):

    barLength, status = 20, ""
    progress = progress / total

    if progress >= 1:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\rCategory {} processing [{}] {:.0f}% {}".format(category,"#" * block + "-" * (barLength - block), round(progress * 100, 0),status)
    stdout.write(text)
    stdout.flush()
