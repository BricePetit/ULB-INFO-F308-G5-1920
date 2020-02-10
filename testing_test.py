from methods import *

def predict(clf,test_fnames,class_names,cluster_model):

    confusion_matrix = np.zeros((len(class_names),len(class_names)), dtype=int)
    testing_data, testing_labels = [], np.array([])
    all_des = []

    for category in test_fnames:
        files = test_fnames[category]
        n = 20#len(files)
        for i in range(n):
            progressBar(n,i+1,category)
            file_name = files[i]
            img = cv2.imread(file_name)
            if img is not None:
                gray = to_gray(img,True)
                """
                Redimensionner pour que nous soyons toujours en train de comparer des images
                de même taille.
                """
                kp, des = gen_sift_features(gray)

                if len(kp) > 0:
                    for descriptor in des:
                        all_des.append(descriptor)

                    testing_data.append(des)
                    testing_labels = np.append(testing_labels,category)

                    a = cluster_model.get_img_clustered_words([des])
                    b = cluster_model.get_img_bow_hist(a)

                    class_prediction = clf.predict(b)
                    """
                    La probabilité de la classe la plus élevée seulement.
                    """
                    svm_prediction = clf.predict_proba(b)
                    """
                    Montrer toutes les probabilités pour chaque classe.
                    """
                    category_index = class_names.index(category)
                    predict_category_index = class_names.index(class_prediction)
                    confusion_matrix[category_index][predict_category_index] += 1
                    #show(img,gray,kp,class_prediction,svm_prediction,category)


    a = cluster_model.get_img_clustered_words(testing_data)
    X = cluster_model.get_img_bow_hist(a)

    accuracy = clf.score(X, testing_labels)
    #show_confusion_matrix(class_names,confusion_matrix,accuracy)
    return accuracy
