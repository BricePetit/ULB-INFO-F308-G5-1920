from methods import *

def train(train_fnames):

    training_data, training_labels = [], np.array([])
    all_descriptors = []

    for category in train_fnames:
        files = train_fnames[category]

        n = 10#len(files)
        for i in range(n):
            progressBar(n,i+1,category)
            file_name = files[i]
            img = cv2.imread(file_name)
            if img is not None:
                gray = to_gray(img,True)
                """
                En comparant des images de même taille, nous pouvons aussi faire des
                images plus grandes ou plus petites pour plus de précision ou plus
                de vitesse.
                """
                kp, des = gen_sift_features(gray)

                if len(kp) > 0:
                    """
                    Découper le vecteur pour que tous soient de la même taille.
                    """
                    for descriptor in des:
                        all_descriptors.append(descriptor)
                    training_data.append(des)
                    """
                    Nous devons le concaténer avec la liste complète des fonctionnalités
                    extraites de chaque image.
                    """
                    training_labels = np.append(training_labels,category)

    return np.array(all_descriptors),training_data,training_labels
