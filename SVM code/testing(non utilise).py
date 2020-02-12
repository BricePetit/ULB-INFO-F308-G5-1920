from methods import *

def predict(clf,test_fnames,class_names):

    confusion_matrix = np.zeros((len(class_names),len(class_names)), dtype=int)
    testing_data, testing_labels = np.array([]), np.array([])

    for category in test_fnames:
        files = test_fnames[category]
        n = 5#len(files)
        for i in range(n):
            progressBar(n,i+1,category)
            file_name = files[i]
            img = cv2.imread(file_name)
            if img is not None:
                gray = to_gray(img,size)
                """
                Redimensionner pour que nous soyons toujours en train de comparer des images
                de même taille.
                """
                kp, des = gen_sift_features(gray)

                if len(kp) > 0:

                    if len(kp) < min_kp:
                        """
                        S'assurer que nous avons au moins min_kp points clés à analyser.
                        """
                        kp, des = duplicate_kp(kp,des,min_kp)

                    np.random.shuffle(des)
                    """
                    Mélanger le vecteur pour obtenir un échantillon représentatif.
                    """
                    des = des[0:min_kp, :].reshape(1, 128 * min_kp)
                    """
                    Découper le vecteur pour que tous soient de la même taille.
                    """
                    if len(testing_data) == 0:
                        testing_data = des
                    else:
                        testing_data = np.concatenate((testing_data, des), axis=0)

                    testing_labels = np.append(testing_labels,category)

                    class_prediction = clf.predict(des)
                    """
                    La probabilité de la classe la plus élevée seulement.
                    """
                    svm_prediction = clf.predict_proba(des)
                    """
                    Montrer toutes les probabilités pour chaque classe.
                    """
                    category_index = class_names.index(category)
                    predict_category_index = class_names.index(class_prediction)
                    confusion_matrix[category_index][predict_category_index] += 1
                    #show(img,gray,kp,class_prediction,svm_prediction,category)

    score = clf.score(testing_data, testing_labels)
    show_confusion_matrix(class_names,confusion_matrix,score)
