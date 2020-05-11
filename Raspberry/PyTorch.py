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

if __name__ == "__main__":
    start = time.time()

    categories = ["Blanc", "Bleu", "Jaune", "Orange", "Verre"]
    
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

    blanc = 0
    correct_blanc = 0
    for f in listdir("dataset/Blanc"):
        file = join("dataset/Blanc", f)
        if isfile(file):
            blanc += 1
            img = Image.open(file)
            img = preprocess(img)
            img = torch.unsqueeze(img, 0)
            res = model_pytorch(img)
            value, index = torch.max(res, 1)
            if categories[0] == categories[int(index)]:
                correct_blanc += 1
    
    bleu = 0
    correct_bleu = 0
    for f in listdir("dataset/Bleu"):
        file = join("dataset/Bleu", f)
        if isfile(file):
            bleu += 1
            img = Image.open(file)
            img = preprocess(img)
            img = torch.unsqueeze(img, 0)
            res = model_pytorch(img)
            value, index = torch.max(res, 1)
            if categories[1] == categories[int(index)]:
                correct_bleu += 1
        
    jaune = 0
    correct_jaune = 0
    for f in listdir("dataset/Jaune"):
        file = join("dataset/Jaune", f)
        if isfile(file):
            jaune += 1
            img = Image.open(file)
            img = preprocess(img)
            img = torch.unsqueeze(img, 0)
            res = model_pytorch(img)
            value, index = torch.max(res, 1)
            if categories[2] == categories[int(index)]:
                correct_jaune += 1

    orange = 0
    correct_orange = 0
    for f in listdir("dataset/Orange"):
        file = join("dataset/Orange", f)
        if isfile(file):
            orange += 1
            img = Image.open(file)
            img = preprocess(img)
            img = torch.unsqueeze(img, 0)
            res = model_pytorch(img)
            value, index = torch.max(res, 1)
            if categories[3] == categories[int(index)]:
                correct_orange += 1

    verre = 0
    correct_verre = 0
    for f in listdir("dataset/Verre"):
        file = join("dataset/Verre", f)
        if isfile(file):
            verre += 1
            img = Image.open(file)
            img = preprocess(img)
            img = torch.unsqueeze(img, 0)
            res = model_pytorch(img)
            value, index = torch.max(res, 1)
            if categories[4] == categories[int(index)]:
                correct_verre += 1

    end_prediction = time.time()

    total = 0

    start_average_predict = time.time()
    img = Image.open("dataset/Blanc/Blanc1.jpg")
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    res = model_pytorch(img)
    value, index = torch.max(res, 1)
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict

    start_average_predict = time.time()
    img = Image.open("dataset/Bleu/Bleu1.jpg")
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    res = model_pytorch(img)
    value, index = torch.max(res, 1)
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict

    start_average_predict = time.time()
    img = Image.open("dataset/Jaune/Jaune1.jpg")
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    res = model_pytorch(img)
    value, index = torch.max(res, 1)
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict

    start_average_predict = time.time()
    img = Image.open("dataset/Orange/Orange1.jpg")
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    res = model_pytorch(img)
    value, index = torch.max(res, 1)
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict

    start_average_predict = time.time()
    img = Image.open("dataset/Verre/Verre1.jpg")
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    res = model_pytorch(img)
    value, index = torch.max(res, 1)
    end_average_predict = time.time()
    
    total += end_average_predict - start_average_predict
    
    end = time.time()

    print("Temps total de l'execution du script : ", end-start)
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