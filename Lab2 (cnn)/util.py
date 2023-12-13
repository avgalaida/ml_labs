import os
import cv2
import torch
from torchvision import transforms
from matplotlib import pyplot as plt


def count_parameters(model):
    return '{:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad))


def show(loader, n, m, classes):
    images, labels = next(iter(loader))

    cols, rows = n, m
    figure = plt.figure(figsize=(10, 8))
    images = torch.clamp(images, 0, 1)
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        plt.title(classes[labels[i - 1]])
        plt.imshow(images[i - 1].permute(1, 2, 0))
        plt.axis("off")


def test(model, loader, classes):
    device = torch.device('mps')

    images, labels = next(iter(loader))

    images = images.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)

    predicted_classes = torch.argmax(outputs, dim=1)
    labels = labels.numpy()
    predicted_classes = predicted_classes.cpu().numpy()

    images = torch.clamp(images, 0, 1)

    plt.figure(figsize=(15, 15))
    for i in range(len(images)):
        if i >= 25:
            break
        plt.subplot(5, 5, i + 1)
        plt.imshow((images[i]).permute(1, 2, 0).cpu().numpy())

        pred = classes[predicted_classes[i]]
        real = classes[labels[i]]
        color = 'green' if pred == real else 'red'

        plt.title(f'{pred}' + '\n' + f'{real}', color=color)
        plt.axis('off')

    plt.show()


def plot(history):
    train_losses = history['train']['losses']
    train_accs = history['train']['accs']
    valid_losses = history['valid']['losses']
    valid_accs = history['valid']['accs']

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, valid_losses, 'g', label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b', label='Training accuracy')
    plt.plot(epochs, valid_accs, 'g', label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
