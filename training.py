from methods import *

#Feature extraction and training

def train(train_file_paths):

    training_data = np.array([])
    training_labels = np.array([])

    for key in train_file_paths:
        category, directory_path = key[1], key[2]
        files_list = train_file_paths[key]

        for i in range(len(files_list)):
            progressBar(len(files_list),i+1,category)
            file_name = directory_path + files_list[i]
            #print(file_name)
            #print("Category = " + str(category))

            img = cv2.imread(file_name)
            if img is not None:

                gray = to_gray(img,size)
                """
                En comparant des images de même taille, nous pouvons aussi faire des
                images plus grandes ou plus petites pour plus de précision ou plus
                de vitesse.
                """
                kp, des = gen_sift_features(gray)

                if len(kp) > 0:
                    if len(kp) < 100:
                        """
                        S'assurer que nous avons au moins 100 points clés à analyser.
                        """
                        kp,des = duplicate_kp(kp,des)

                    des = des.astype(np.float64)
                    """
                    Transformer les données en flottant et mélanger tous les points clés
                    pour obtenir un échantillonnage aléatoire de chaque image.
                    """
                    np.random.shuffle(des)
                    des = des[0:100,:]
                    """
                    Découper le vecteur pour que tous soient de la même taille.
                    """
                    vector_data = des.reshape(1,12800)
                    list_data = vector_data.tolist()
                    """
                    Nous devons le concaténer avec la liste complète des fonctionnalités
                    extraites de chaque image.
                    """
                    if len(training_data) == 0:
                        training_data = np.append(training_data, vector_data)
                        training_data = training_data.reshape(1,12800)
                    else:
                        training_data   = np.concatenate((training_data, vector_data), axis=0)

                    training_labels = np.append(training_labels,category)

    return training_data,training_labels
