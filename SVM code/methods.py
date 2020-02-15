import cv2
from sys import stdout

def gen_sift_features(image,resize):
    """
    Create SIFT model and extract images features
    kp is the keypoints
    desc is the SIFT descriptors, they're 128-dimensional vectors
    """
    sift = cv2.xfeatures2d.SIFT_create()

    if not resize:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.resize(image, resize)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, desc = sift.detectAndCompute(gray_image, None)
    return kp, desc

def features_extraction(fnames, data, labels, resize):
    print("The start of features extraction")
    for category in fnames:
        files = fnames[category]
        n = len(files)
        for i in range(n):
            progressBar(n,i+1,category)
            file_name = files[i]
            img = cv2.imread(file_name)
            if img is not None:
                kp, des = gen_sift_features(img,resize)
                """
                En comparant des images de même taille, nous pouvons aussi faire
                des images plus grandes ou plus petites pour plus de précision
                ou plus de vitesse.
                """
                if len(kp) > 0:
                    data.append(des)
                    labels.append(category)
    print("The end of the extraction")

def progressBar(total, progress, category):

    barLength, status = 20, ""
    progress = progress / total

    if progress >= 1:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\rProcessing of {} items [{}] {:.0f}% {}".format(str(total)+ " " + category,"#" * block + "-" * (barLength - block), round(progress * 100, 0),status)
    stdout.write(text)
    stdout.flush()
