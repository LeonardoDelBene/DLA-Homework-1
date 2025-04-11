import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split
from myNetworks import CNN, ResNet
from pipeline import train_validate, evaluate_model


def cnn(train_loader, val_loader, test_loader, batch):
    lr = 0.0001
    wd = 0.01
    num_blocks = [2, 2, 2]
    hidden_channels = [128, 256, 512]
    input_channels = 3
    num_class = 10
    dim_input = 32
    num_epochs = 20
    batch_size = batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    num_models = 1

    wandb.login(key="bfa1df1c98b555b96aa3777a18a6e8ca9b082d53")

    for i in range(num_models):
        depth = len(num_blocks) + 1  # len(num_layers) è il numero di conv nele transizioni più la conv iniziale +1 dovuto al fc
        for _ in range(len(num_blocks)):
            depth += num_blocks[_]

        wandb.init(project="Homework-1-CNN", name="CNN" + "/" + str(depth) + "/" + str(hidden_channels[-1]))
        wandb.config.update({
            "learning_rate": lr,
            "weight_decay": wd,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "architecture": "CNN",
            "hidden_layers": hidden_channels[1:-1]
        })

        model = CNN(num_blocks, input_channels, hidden_channels, num_class, dim_input).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        train_validate(model, num_epochs, train_loader, val_loader, opt, criterion, device, "cnn_cifar10.pth")

        evaluate_model(model, test_loader, criterion, device)

        for j in range(len(num_blocks)):
            num_blocks[j] *= 3


def resnet(train_loader, val_loader, test_loader, batch):
    lr = 0.0001
    wd = 0.01
    num_layers = [1, 1, 1, 1]  # ci sono 4 macroblocchi composti da x sottoblocchi, ogni sottoblocco ha 2 conv
    channels = [128, 256, 512, 1024]
    in_channels = 3
    num_class = 10
    num_epochs = 20
    batch_size = batch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_models = 3

    wandb.login(key="bfa1df1c98b555b96aa3777a18a6e8ca9b082d53")

    for i in range(num_models):
        depth = len(num_layers) + 1  # len(num_layers)-1 è il numero di conv nele transizioni + la conv iniziale + fc
        for _ in range(len(num_layers)):
            depth += num_layers[_] * 2  # ogni blocco ha due conv

        wandb.init(project="Homework-1-CNN", name="RESNET" + "/" + str(depth) + "/" + str(channels[-1]))
        wandb.config.update({
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "architecture": "RESNET",
            "hidden_layers": channels[1:-1]
        })

        model = ResNet(channels, num_layers, in_channels, num_class).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        train_validate(model, num_epochs, train_loader, val_loader, opt, criterion, device, "resnet_cifar10.pth")

        evaluate_model(model, test_loader, criterion, device)

        if (i != 2):
            for j in range(len(num_layers)):
                num_layers[j] += 1
        else:
            for j in range(len(num_layers)):
                num_layers[j] += 5

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    cnn(train_loader, val_loader, test_loader, batch_size)
    resnet(train_loader, val_loader, test_loader, batch_size)


