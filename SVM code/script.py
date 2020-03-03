from Model import *
import joblib

SIZE = (384, 512)
k = 1000
#------------------------------------------------------------------------------

def features_extraction(image_name, resize):

    gray_image = cv2.imread(image_name,0)
    if gray_image is not None:
        sift = cv2.xfeatures2d.SIFT_create()
        if resize:
            gray_image = cv2.resize(gray_image, resize, interpolation=cv2.INTER_AREA)

        kp, desc = sift.detectAndCompute(gray_image, None)

        if len(kp) > 0:
            return desc
    return "ERROR"

cluster_model = ClusterModel()
cluster_model.load_model('cluster-model/1000-cluster.pkl')
clf = ImageClassifierModel()
clf.load_model('classification-model/1000-classification.pkl')

def predict(image_name,cluster_model,clf) :
    image_desc = features_extraction(image_name, resize = SIZE)
    img_clustered_words = cluster_model.get_img_clustered_words([image_desc])
    X = cluster_model.get_img_bow_hist(img_clustered_words,k)
    y_pred = clf.clf.predict(X)
    return y_pred[0]

predict("../data/Blanc/plastic349.jpg",cluster_model,clf)
