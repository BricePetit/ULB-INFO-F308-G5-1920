from Model import *
import joblib

SIZE = (384, 512)
k = 1000
#------------------------------------------------------------------------------

def features_extraction(image_name, model, resize):

    gray_image = cv2.imread(image_name,0)
    if gray_image is not None:
        if model == "SIFT":
            mdl = cv2.xfeatures2d.SIFT_create()
        elif model == "SURF":
            mdl = cv2.xfeatures2d.SURF_create(extended = True, hessianThreshold = 400)
        elif model == "ORB":
            mdl = cv2.ORB_create(1000)
        if resize:
            gray_image = cv2.resize(gray_image, resize, interpolation=cv2.INTER_AREA)

        kp, desc = mdl.detectAndCompute(gray_image, None)

        if len(kp) > 0:
            return desc
    return "ERROR"

cluster_model = ClusterModel()
clf = ImageClassifierModel()
model = "ORB"
cluster_model.load_model('cluster-model/1000-{}.pkl'.format(model))
clf.load_model('classification-model/1000-SVM-{}.pkl'.format(model))

def predict(image_name,cluster_model,clf) :
    image_desc = features_extraction(image_name, model, resize = SIZE)
    img_clustered_words = cluster_model.get_img_clustered_words([image_desc])
    X = cluster_model.get_img_bow_hist(img_clustered_words,k)
    y_pred = clf.clf.predict(X)
    return y_pred[0]

predict("../data/Blanc/Blanc349.jpg",cluster_model,clf)
