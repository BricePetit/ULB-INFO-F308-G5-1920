from torchvision import models,transforms
import torch.nn as nn
import torch
from PIL import Image
from torchvision.transforms.transforms import Normalize
from torchvision.datasets import ImageFolder
import os
from torch.utils.data.dataloader import DataLoader


def get_pytorch_model():
    model = models.vgg19(pretrained=True)
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Linear(512, 343),
        nn.ReLU(),
        nn.Linear(343, 174),
        nn.ReLU(),
        nn.Linear(174, 5),
        nn.Softmax(dim=1))
    #m = nn.AvgPool2d(3, stride=2)
    #m = nn.AvgPool2d((3, 2), stride=(2, 1))
    #input = torch.randn(20, 16, 50, 32)
    #output = m(input)
    return model


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image

if __name__ == "__main__":
    model = get_pytorch_model()
    preprocess = transforms.Compose([transforms.Resize(256), transforms.ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
"""
    train_data = ImageFolder(root = "../dataset", transform=preprocess)
    # val_data = ImageFolder(root = "../dataset", transform=preprocess)

    data_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)

    train_loader = iter(data_loader)

    x,y = next(train_loader)
"""
    # inférence avec images 
    # img = Image.open("../dataset/Blanc/Blanc1.jpg")
    # img = preprocess(img)
    # img = torch.unsqueeze(img, 0)

    # model = model.eval()
    # probs = model(img)
    # print(probs)
    # values, index = torch.max(probs, 1)
    
    # print(str(int(index.numpy())))
    

    # loader = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

    # img = image_loader("../dataset/Blanc/Blanc1.jpg")

    # print(model(img).data)

    # torch.save(model, "test.test")
    # model = torch.load("test.test")