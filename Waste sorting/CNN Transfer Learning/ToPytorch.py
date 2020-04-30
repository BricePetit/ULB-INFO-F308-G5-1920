from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
from PIL import Image
from torchvision.transforms.transforms import Normalize
from torchvision.datasets import ImageFolder
import os
from torch.utils.data.dataloader import DataLoader
import copy
import time
from torch import device, optim

# Check accuracy on training & test to see how good our model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()


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
    for param in model.parameters():
        param.requires_grade = False
    return model


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image

if __name__ == "__main__":
    model = get_pytorch_model()
    preprocess = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
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

    num_classes = 10 
    learning_rate = 1e-3
    batch_size = 1024
    num_epochs = 5

    train_dataset = ImageFolder(root = "../dataset", transform=preprocess)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
        losses = []
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            
            losses.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent or adam step
            optimizer.step()
        
    check_accuracy(train_loader, model)
    
    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}')
    

    # loader = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

    # img = image_loader("../dataset/Blanc/Blanc1.jpg")

    # print(model(img).data)

    # torch.save(model, "test.test")
    # model = torch.load("test.test")