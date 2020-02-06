from methods import *

def train(train_fnames):

    training_data, training_labels = np.array([]), np.array([])

    for category in train_fnames:
        files = train_fnames[category]

        n = 10#len(files)
        for i in range(n):
            progressBar(n,i+1,category)
            file_name = files[i]
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
                    if len(kp) < min_kp:
                        """
                        S'assurer que nous avons au moins min_kp points clés à analyser.
                        """
                        kp,des = duplicate_kp(kp,des,min_kp)

                    """
                    Transformer les données en flottant et mélanger tous les points clés
                    pour obtenir un échantillonnage aléatoire de chaque image.
                    """
                    np.random.shuffle(des)
                    """
                    Découper le vecteur pour que tous soient de la même taille.
                    """
                    des = des[0:min_kp,:].reshape(1,128 * min_kp)
                    """
                    Nous devons le concaténer avec la liste complète des fonctionnalités
                    extraites de chaque image.
                    """
                    if len(training_data) == 0:
                        training_data = des
                    else:
                        training_data = np.concatenate((training_data, des), axis=0)

                    training_labels = np.append(training_labels,category)

    return training_data,training_labels
