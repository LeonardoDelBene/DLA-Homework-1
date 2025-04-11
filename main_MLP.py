import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.optim import Adam, AdamW
from myNetworks import MLP, Residual_MLP
from pipeline import train_validate, evaluate_model

def mlp(train_loader, val_loader, test_loader,batch):
    lr = 0.0001
    input_size = 28 * 28
    hidden_size = 64
    output_size = 10
    num_epochs = 50
    batch_size = batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_models = 2
    hidden_layers_list = [34,44]

    wandb.login(key="bfa1df1c98b555b96aa3777a18a6e8ca9b082d53")

    for i in range(num_models):
        for j in range(hidden_layers_list[i]):
            layer_size = [input_size] + [hidden_size] * j + [output_size]
        depth = len(layer_size) - 1

        wandb.init(project="Homework-1-MLP", name="MLP-Training / " + str(depth))
        wandb.config.update({
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "architecture": "Residual MLP",
            "depth": depth,
            "hidden_size": hidden_size
        })

        model = MLP(layer_size).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        train_validate(model, num_epochs, train_loader, val_loader, opt, criterion, device, "mlp_mnist.pth")

        evaluate_model(model, test_loader, criterion, device)


def res_mlp(train_loader, val_loader, test_loader,batch):
    lr = 0.0001
    num_blocks = 6
    hidden_size = 256
    input_size = 28 * 28
    num_epochs = 50
    batch_size = batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_models = 2
    add_blocks = 5

    wandb.login(key="bfa1df1c98b555b96aa3777a18a6e8ca9b082d53")

    for i in range(num_models):

        depth = num_blocks * 2 + 2

        wandb.init(project="Homework-1-MLP", name="Residual-MLP-Training / " + str(num_blocks) + "-" + str(depth))
        wandb.config.update({
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "architecture": "Residual MLP",
            "hidden_layers": hidden_size,
            "num_blocks": num_blocks
        })

        model = Residual_MLP(num_blocks, input_size, hidden_size).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_validate(model, num_epochs, train_loader, val_loader, opt, criterion, device, "residual_mlp_mnist.pth")

        evaluate_model(model, test_loader, criterion, device)

        num_blocks += add_blocks


if __name__ == "__main__":
    mnist_path = "/Users/leonardodelbene/Datasets/mnist.npz"
    
    if not os.path.exists(mnist_path):
        raise FileNotFoundError(f"Il file MNIST non è stato trovato in {mnist_path}")

    data = np.load(mnist_path)
    
    train_images = data['x_train']
    train_labels = data['y_train']
    test_images = data['x_test']
    test_labels = data['y_test']

    train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0
    test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1) / 255.0
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Dataset caricato con successo!")

    mlp(train_loader, val_loader, test_loader, batch_size)
    res_mlp(train_loader, val_loader, test_loader, batch_size)

