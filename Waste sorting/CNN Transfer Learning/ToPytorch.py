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
from torch.optim import lr_scheduler
import shutil
import random

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
    # model.classifier[6] = nn.Sequential(
    #     nn.Linear(512, 100),
    #     nn.ReLU(),
    #     nn.Linear(100, 5))
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
    # model = get_pytorch_model()
    # preprocess = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
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

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_classes = 10 
    learning_rate = 1e-3
    batch_size = 1024
    num_epochs = 100

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
    """

    # loader = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

    # img = image_loader("../dataset/Blanc/Blanc1.jpg")

    # print(model(img).data)

    # torch.save(model, "test.test")
    # model = torch.load("test.test")
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def split(train_datagen, test_datagen, directory, test_split=0.2, fraction_dataset=1):
        try:
            shutil.rmtree("datasets")
        except:
            pass

        os.mkdir("datasets")
        os.mkdir("datasets/train")
        os.mkdir("datasets/val")

        for sub_dir in os.listdir(directory):
            images = os.listdir(directory + "/" + sub_dir)
            random.shuffle(images)
            images = images[:round(fraction_dataset * len(images))]
            test_images = images[:round(test_split * len(images))]
            train_images = images[round(test_split * len(images)):]

            os.mkdir("datasets/val/" + sub_dir)

            for i in test_images:
                os.link(directory + "/" + sub_dir + "/" + i, "datasets/val/" + sub_dir + "/" + i)

            os.mkdir("datasets/train/" + sub_dir)

            for i in train_images:
                os.link(directory + "/" + sub_dir + "/" + i, "datasets/train/" + sub_dir + "/" + i)

        data_transforms = {
        'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
        return data_transforms

    data_dir = '../dataset'
    data_transforms = split()
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 5)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)