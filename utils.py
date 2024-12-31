import os
import matplotlib.pyplot as plt
import re


def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def extract_classes(dataset):
    class_set = set()
    for i in range(len(dataset.dataset)):
        _, data = dataset.dataset[i]
        suffix = data["suffix"]
        classes = re.findall(r"([a-zA-Z0-9 ]+ of [a-zA-Z0-9 ]+)<loc_\d+>", suffix)
        class_set.update(classes)
    return sorted(class_set)
