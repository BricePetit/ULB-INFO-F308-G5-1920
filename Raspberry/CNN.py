"""
INFO-F308 : Projet d'année 3 2019-2020

Tri des déchets, élémentaire, n'est-ce pas (v 2.0) ?

Script pour le benchmark de Keras sur le Raspberry Pi 4

Pour changer de backend sous keras, il faut changer "backend" dans $HOME/.keras/keras.json
Sous windows, il faut remplacer $HOME par %USERPROFILE%
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
par celui désiré, "tensorflow", "theano" ou "cntk"

"""

import time
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from os import listdir
from os.path import isfile, join

def browse(colorToTest):
    for f in listdir(path):
        file = join(path, f)
        if isfile(file):
            color[colorToTest] += 1
            if cat[0] == cat[predictKerasCNN(file)]:
                correct_color[colorToTest] += 1

def predictKerasCNN(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.expand_dims(img, axis=0)
    result = model_tensorflow.predict([img])
    return cat[np.argmax(result[0])]

if __name__ == "__main__":
    start = time.time()

    cat = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]

    start_load_model = time.time()
    model_tensorflow = load_model('Model.model')
    end_load_model = time.time()
    
    start_prediction = time.time()
    color = [0,0,0,0,0]
    correct_color = [0,0,0,0,0]
    for colorToTest in range(len(cat)):
        path = "dataset/" + cat[colorToTest]
        browse(colorToTest)
    end_prediction = time.time()

    total = 0
    for colorToTest in range(len(cat)):
        start_average_predict = time.time()
        file = "dataset/" + cat[colorToTest] + "/" + cat[colorToTest] +"1.jpg"
        predictKerasCNN(file)
        end_average_predict = time.time()
        total += end_average_predict - start_average_predict
    
    end = time.time()

    print("Temps total de l'execution du script : ", end-start)
    print("Temps du chargement du model : ", end_load_model-start_load_model)
    print("Temps des prédictions : ", end_prediction-start_prediction)
    print("Temps de prédiction moyen pour une image : ", total/5)

    print("Précision pour la poubelle blanche : ", (correct_color[0]/color[0]) * 100, "%")
    print("Précision pour la poubelle bleue : ", (correct_color[1]/color[1]) * 100, "%")
    print("Précision pour la poubelle jaune : ", (correct_color[2]/color[2]) * 100, "%")
    print("Précision pour la poubelle orange : ", (correct_color[3]/color[3]) * 100, "%")
    print("Précision pour le verre : ", (correct_color[4]/color[4]) * 100, "%")
    print("Précision totale : ", ((correct_color[0]+correct_color[1]+correct_color[2]+correct_color[3]+correct_color[4])/\
        (color[0]+color[1]+color[2]+color[3]+color[4])) *100, "%")