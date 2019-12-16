from methods import *

#Predict the whole Data Set

def predict(clf,test_fnames,class_names):

    score_matrix = np.zeros((len(class_names),len(class_names)), dtype=int)
    score = 0

    for i in range(len(test_fnames)):

        print(i)
        file_name, file_class = test_fnames[i][1], test_fnames[i][0]
        img = cv2.imread(file_name)
        if img is not None:
            gray = to_gray(img,size)
            """
            Redimensionner pour que nous soyons toujours en train de comparer des images
            de même taille.
            """
            kp, des = gen_sift_features(gray)

            if len(kp) > 0:

                if len(kp) < 100:
                    """
                    S'assurer que nous avons au moins 100 points clés à analyser.
                    """
                    kp, des = duplicate_kp(kp,des)

                np.random.shuffle(des)
                """
                Mélanger le vecteur pour obtenir un échantillon représentatif.
                """
                des = des[0:100, :]
                """
                Découper le vecteur pour que tous soient de la même taille.
                """
                vector_data = des.reshape(1, 12800)

                class_prediction = class_names[int(clf.predict(vector_data))]
                """
                Classe de probabilité la plus élevée seulement.
                """
                svm_prediction = clf.predict_proba(vector_data)
                """
                Montrer toutes les probabilités pour chaque classe.
                """
                score += 1 if class_prediction == file_class else 0
                category = class_names.index(file_class)
                predict_category = class_names.index(class_prediction)
                score_matrix[category][predict_category] += 1
                #show(img,gray,kp,class_prediction,svm_prediction,file_class)

    show_matrix_prediction(class_names,score_matrix,score)
