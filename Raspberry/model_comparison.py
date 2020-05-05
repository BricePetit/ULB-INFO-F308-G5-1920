"""
INFO-F308 : Projet d'année 3 2019-2020

Tri des déchets, élémentaire, n'est-ce pas (v 2.0) ?

Script pour le benchmark sur le Raspberry Pi 4

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
from keras.models import load_model
from keras.preprocessing import image
import torch

def predictKerasCNN(img_path):
    categories = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.expand_dims(img, axis=0)
    result = model_tensorflow.predict([img])
    return categories[np.argmax(result[0])]

if __name__ == "__main__":
    model_tensorflow = load_model('Model.model')
    # res = predictKerasCNN("dechet.jpg")
    # categories = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]
    # index = categories.index(res)

    model_pytorch = torch.load("ModelPyTorch.pt")
    model_pytorch.eval()