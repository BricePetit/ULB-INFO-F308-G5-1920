"""
INFO-F308 : Projet d'année 3 2019-2020

Tri des déchets, élémentaire, n'est-ce pas (v 2.0) ?

Script pour le benchmark de Keras sur le Raspberry Pi 4

Pour changer de backend sous keras, il faut changer "backend" dans $HOME/.keras/keras.json
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

def predictKerasCNN(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.expand_dims(img, axis=0)
    result = model_tensorflow.predict([img])
    return categories[np.argmax(result[0])]

if __name__ == "__main__":
    start = time.time()

    categories = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]

    start_load_model = time.time()
    model_tensorflow = load_model('Model.model')
    end_load_model = time.time()
    
    start_prediction = time.time()

    blanc = 0
    correct_blanc = 0
    for f in listdir("dataset/Blanc"):
        file = join("dataset/Blanc", f)
        if isfile(file):
            blanc += 1
            if categories[0] == predictKerasCNN(file):
                correct_blanc += 1
    
    bleu = 0
    correct_bleu = 0
    for f in listdir("dataset/Bleu"):
        file = join("dataset/Bleu", f)
        if isfile(file):
            bleu += 1
            if categories[1] == predictKerasCNN(file):
                correct_bleu += 1
        
    jaune = 0
    correct_jaune = 0
    for f in listdir("dataset/Jaune"):
        file = join("dataset/Jaune", f)
        if isfile(file):
            jaune += 1
            if categories[2] == predictKerasCNN(file):
                correct_jaune += 1

    orange = 0
    correct_orange = 0
    for f in listdir("dataset/Orange"):
        file = join("dataset/Orange", f)
        if isfile(file):
            orange += 1
            if categories[3] == predictKerasCNN(file):
                correct_orange += 1

    verre = 0
    correct_verre = 0
    for f in listdir("dataset/Verre"):
        file = join("dataset/Verre", f)
        if isfile(file):
            verre += 1
            if categories[4] == predictKerasCNN(file):
                correct_verre += 1
    end_prediction = time.time()

    total = 0

    start_average_predict = time.time()
    predictKerasCNN("dataset/Blanc/Blanc1.jpg")
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict

    start_average_predict = time.time()
    predictKerasCNN("dataset/Bleu/Bleu1.jpg")
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict

    start_average_predict = time.time()
    predictKerasCNN("dataset/Jaune/Jaune1.jpg")
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict

    start_average_predict = time.time()
    predictKerasCNN("dataset/Orange/Orange1.jpg")
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict

    start_average_predict = time.time()
    predictKerasCNN("dataset/Verre/Verre1.jpg")
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict
    
    end = time.time()

    print("Temps total de l'exécution du script : ", end-start)
    print("Temps du chargement du model : ", end_load_model-start_load_model)
    print("Temps des prédictions : ", end_prediction-start_prediction)
    print("Temps de prédiction moyen pour une image : ", total/5)
    
    print("Précision pour la poubelle blanche : ", (correct_blanc/blanc) * 100, "%")
    print("Précision pour la poubelle bleu : ", (correct_bleu/bleu) * 100, "%")
    print("Précision pour la poubelle jaune : ", (correct_jaune/jaune) * 100, "%")
    print("Précision pour la poubelle orange : ", (correct_orange/orange) * 100, "%")
    print("Précision pour le verre : ", (correct_verre/verre) * 100, "%")
    print("Précision totale : ", ((correct_blanc+correct_bleu+correct_jaune+correct_orange+correct_verre)/\
        (blanc+bleu+jaune+orange+verre)) *100, "%")