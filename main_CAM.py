import ssl
from wandb import summary
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import torchvision
from myNetworks import CNN_CAM, CAM
from pipeline import train_validate, evaluate_model
import torchvision.models as models
import random
import os
import wandb



def overlay_cam(image_path, cam, alpha=0.5):
    # Carica l'immagine e ridimensionala alle dimensioni della CAM
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

    return img, superimposed_img


def preprocess_image(image_path,transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Aggiunge la dimensione batch
    return image


def train_model(train_loader, val_loader, test_loader,batch,model_cnn, name_run, filename,checkpoint=None):
    lr = 0.0001
    wd = 0.01
    num_epochs = 100
    batch_size = batch


    wandb.login(key="bfa1df1c98b555b96aa3777a18a6e8ca9b082d53")
    wandb.init(project="Homework-1-CNN-CAM", name=name_run)
    wandb.config.update({
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "architecture": "CNN",
    })

    model = model_cnn
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    train_validate(model, num_epochs, train_loader, val_loader, opt, criterion, device, filename,checkpoint)

    evaluate_model(model, test_loader, criterion, device)

    return model


def fine_tuning(model, train_loader, val_loader, num_epochs, learning_rate, device, filename, name_run):
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    wandb.login(key="bfa1df1c98b555b96aa3777a18a6e8ca9b082d53")
    # Inizializza il run su Weights & Biases
    wandb.init(project="Homework-1-CNN-CAM", name=name_run, config={
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": learning_rate,
        "model": "resnet18",
        "optimizer": optimizer,
        "loss": criterion
    })
    train_validate(model, num_epochs, train_loader, val_loader, optimizer, criterion, device, filename)
    evaluate_model(model, val_loader, criterion, device)
    return model


# Main part
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = "/tmp/DLA/homework1/Lab1/.venv/imagenette-160"
    wnid_to_label = {
        "n01440764": "Tench",
        "n02102040": "English springer",
        "n02979186": "Cassette player",
        "n03000684": "Chain saw",
        "n03028079": "Church",
        "n03394916": "French horn",
        "n03417042": "Garbage truck",
        "n03425413": "Gas pump",
        "n03445777": "Golf ball",
        "n03888257": "Parachute"
    }

    transform = transforms.Compose([
        transforms.RandomCrop(64),
        #transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_path}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_path}/val", transform=transform)

    batch_size = 20

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Modello CNN
    num_blocks = [2, 2, 2, 2]
    hidden_channels = [64, 128, 256, 512]
    input_channels = 3
    num_class = 10

    model_cnn = CNN_CAM(num_blocks, input_channels, hidden_channels, num_class)
    model_cnn.to(device)
    model_cnn_save_path = "--"  # usa "cnn_imageNette.pth" per ImageNette chackponit a "cnn_imageNette_checkpoint.pth"
    if os.path.exists(model_cnn_save_path):
        print(f"Caricamento del modello salvato da: {model_cnn_save_path}")
        model_cnn.load_state_dict(torch.load(model_cnn_save_path))
    else:
        print(f"Nessun modello salvato trovato. Inizierò l'addestramento da zero.")
        model_cnn = train_model(train_loader, val_loader, val_loader, batch_size, model_cnn,
                                "CNN/" + str(num_blocks) + "/ImageNette", "cnn_imageNette.pth",
                                "cnn_imageNette_checkpoint.pth")

    # Carica ResNet-18 pre-addestrato
    num_classes = 10
    model_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)
    model_resnet = model_resnet.to(device)

    model_resnet_save_path = "resnet_fine_tuned_model.pth"
    if os.path.exists(model_resnet_save_path):
        print(f"Caricamento del modello salvato da: {model_resnet_save_path}")
        model_resnet.load_state_dict(torch.load(model_resnet_save_path))
    else:
        print(f"Nessun modello salvato trovato. Inizierò il fine-tuning da zero.")
        # Esegui il fine-tuning solo sull'ultimo layer FC
        for param in model_resnet.parameters():
            param.requires_grad = False
        for param in model_resnet.fc.parameters():
            param.requires_grad = True
        model_resnet = fine_tuning(model_resnet, train_loader, val_loader, num_epochs=5, learning_rate=0.001,
                                   device=device, filename="resnet_fine_tuned_model.pth",
                                   name_run="resnet_fine_tuning_imagenette")

    imagenette_val_path = f"{data_path}/val"
    classes = [c for c in os.listdir(imagenette_val_path) if
               not c.startswith('.') and os.path.isdir(os.path.join(imagenette_val_path, c))]

    # Configura il layout per il grafico
    fig, axes = plt.subplots(3, 10, figsize=(20, 8))  # Crea una griglia 2x10 per visualizzare 10 classi

    # Mantieni il ciclo originale che carica immagini da ImageNette
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(imagenette_val_path, class_name)

        # Seleziona un'immagine casuale dalla classe
        random_image = random.choice(os.listdir(class_path))
        image_path = os.path.join(class_path, random_image)

        # Recupera la label umana (se presente)
        human_label = wnid_to_label.get(class_name, class_name)

        # Pre-processa l'immagine
        input_tensor = preprocess_image(image_path, transform)
        input_tensor = input_tensor.to(device)

        # Crea la CAM e visualizza l'overlay per il modello ResNet-18
        cam_generator_resnet = CAM(model_resnet, type="resnet")
        cam_resnet = cam_generator_resnet.generate(input_tensor)
        img, superimposed_img_resnet = overlay_cam(image_path, cam_resnet)

        # Crea la CAM e visualizza l'overlay per il modello CNN
        cam_generator_cnn = CAM(model_cnn, type="cnn")
        cam_cnn = cam_generator_cnn.generate(input_tensor)
        _, superimposed_img_cnn = overlay_cam(image_path, cam_cnn)

        # Mostra le immagini originali e con CAM sovrapposta in una griglia
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f"Original ({human_label})")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(superimposed_img_resnet)  # Immagine con CAM sovrapposto ResNet
        axes[1, idx].set_title(f"ResNet CAM ({human_label})")
        axes[1, idx].axis("off")

        axes[2, idx].imshow(superimposed_img_cnn)
        axes[2, idx].set_title(f"CNN CAM ({human_label})")
        axes[2, idx].axis("off")

    plt.tight_layout()
    plt.savefig("cam_visualization.png")
    plt.show()


