from torchvision import models,transforms
import torch.nn as nn
import torch
from PIL import Image


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
    return model


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image


if __name__ == "__main__":
    model = get_pytorch_model()

    loader = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

    img = image_loader("../dataset/Blanc/Blanc1.jpg")

    print(model(img).data)

    # torch.save(model, "test.test")
    # model = torch.load("test.test")
