import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def train_epoch(model, train_loss, correct_train, total_train, train_loader, opt, criterion, device='cpu',):

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        opt.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    return train_loss, correct_train, total_train


def valid_epoch(model, val_loss, correct_val, total_val, val_loader, criterion, device='cpu'):

  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)
      val_loss += loss.item()

      _, predicted = torch.max(outputs, 1)
      correct_val += (predicted == labels).sum().item()
      total_val += labels.size(0)

  return val_loss, correct_val, total_val

def test(model, test_loss, correct_test, total_test, test_loader, criterion, device='cpu'):

  with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

  return test_loss, correct_test, total_test


def train_validate(model, num_epochs, train_loader, val_loader, opt, criterion, device, filename, checkpoint_filename=None):
    best_val_acc = 0.0
    best_val_loss = float('inf')
    start_epoch = 0

    # Carica il checkpoint se esiste
    if checkpoint_filename and os.path.exists(checkpoint_filename):
        start_epoch, best_val_acc, best_val_loss = load_checkpoint(model, opt, checkpoint_filename)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        train_loss, correct_train, total_train = train_epoch(model, train_loss, correct_train, total_train,
                                                             train_loader, opt, criterion, device)

        train_loss /= len(train_loader)
        train_acc = correct_train / total_train * 100

        # Validazione
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        val_loss, correct_val, total_val = valid_epoch(model, val_loss, correct_val, total_val, val_loader, criterion,
                                                       device)

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val * 100

        # Logga i valori su WandB
        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc
        })

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc and val_loss < best_val_loss:
            best_val_acc = max(val_acc, best_val_acc)
            best_val_loss = min(val_loss, best_val_loss)
            torch.save(model.state_dict(), filename)
            print(f"🧠 Modello migliorato, salvato in: {filename}")
            wandb.save(filename)
            if(checkpoint_filename):
                # Salva il checkpoint
                save_checkpoint(model, opt, epoch, best_val_acc, best_val_loss, checkpoint_filename)
                artifact = wandb.Artifact('model_checkpoint', type='model')
                artifact.add_file(checkpoint_filename)
                wandb.log_artifact(artifact)


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, correct_test, total_test = 0.0, 0, 0
    test_loss, correct_test, total_test = test(model, test_loss, correct_test, total_test, test_loader, criterion, device)

    test_loss /= len(test_loader)
    test_acc = correct_test / total_test * 100

    # Logga il test su WandB
    wandb.log({
        "Test Loss": test_loss,
        "Test Accuracy": test_acc
    })

    print(f"🔹 Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    wandb.finish()


def save_checkpoint(model, optimizer, epoch, best_val_acc, best_val_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint salvato in {filename}")

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Checkpoint caricato da {filename}, riprendo dall'epoca {epoch}")
    return epoch, best_val_acc, best_val_loss
