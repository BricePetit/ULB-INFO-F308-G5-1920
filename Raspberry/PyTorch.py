"""
INFO-F308 : Projet d'année 3 2019-2020

Tri des déchets, élémentaire, n'est-ce pas (v 2.0) ?

Script pour le benchmark de PyTorch sur le Raspberry Pi 4
"""

import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.transforms import Normalize
from os import listdir
from os.path import isfile, join

def browse(colorToTest):
    for f in listdir(path):
        file = join(path, f)
        if isfile(file):
            color[colorToTest] += 1
            if cat[0] == cat[predict(file)]:
                correct_color[colorToTest] += 1

def predict(file):
    img = Image.open(file)
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    res = model_pytorch(img)
    value, index = torch.max(res, 1)
    return int(index)


if __name__ == "__main__":
    start = time.time()

    cat = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]
    
    start_load_model = time.time()
    model_pytorch = torch.load("ModelPyTorch.pt", map_location=torch.device('cpu'))
    end_load_model = time.time()
    
    model_pytorch.eval()
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
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
        predict(file)
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