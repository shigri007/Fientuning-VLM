import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoProcessor

# Constants
CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 72
NUM_WORKERS = 0
dataset = 'D:/research proposall/Foggy Super Resolution/visdroneF'
EPOCHS = 100
LR = 5e-6

# %% Model and Processor Setup
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

# JSONLDataset Class to load dataset from JSONL
class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> list:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            return image, entry
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")

# DetectionDataset Class
class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data['prefix']
        suffix = data['suffix']
        return prefix, suffix, image

# DataLoader Setup
def collate_fn(batch):
    prefixes, suffixes, images = zip(*batch)
    inputs = processor(text=list(prefixes), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, suffixes

# Load CLIP Model with LoRA
def load_clip_model_with_lora():
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,  # Rank of LoRA decomposition
        lora_alpha=32,
        lora_dropout=0.1
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model

# Training Loop
def train_model(model, train_loader, val_loader, optimizer, device):
    model.train()
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(EPOCHS):
        running_loss = 0
        correct_predictions = 0
        total_samples = 0

        for prefixes, suffixes, images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device)
            inputs = processor(text=list(prefixes), images=list(images), return_tensors="pt", padding=True).to(device)
            outputs = model(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'])
            logits_per_image = outputs.logits_per_image
            labels = torch.arange(len(logits_per_image), device=device)
            loss = nn.CrossEntropyLoss()(logits_per_image, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions = torch.argmax(logits_per_image, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        epoch_val_loss, precision, accuracy = evaluate_model(model, val_loader, device)
        val_losses.append(epoch_val_loss)
        accuracies.append(accuracy)
        log_metrics(epoch, epoch_train_loss, epoch_val_loss, precision, accuracy)
        print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}")

    plot_losses(train_losses, val_losses)
    plot_accuracy_map(accuracies)

    return train_losses, val_losses, accuracies

# Evaluation Loop
def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for prefixes, suffixes, images in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            inputs = processor(text=list(prefixes), images=list(images), return_tensors="pt", padding=True).to(device)
            outputs = model(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'])
            logits_per_image = outputs.logits_per_image
            labels = torch.arange(len(logits_per_image), device=device)
            loss = nn.CrossEntropyLoss()(logits_per_image, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits_per_image, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    precision = correct_predictions / total_samples
    accuracy = correct_predictions / total_samples

    return avg_loss, precision, accuracy

# Log Metrics
def log_metrics(epoch, train_loss, val_loss, precision, accuracy):
    with open("training_loss.txt", "a") as f_train_loss:
        f_train_loss.write(f"Epoch {epoch+1}: {train_loss:.4f}\n")
    with open("validation_loss.txt", "a") as f_val_loss:
        f_val_loss.write(f"Epoch {epoch+1}: {val_loss:.4f}\n")
    with open("precision.txt", "a") as f_precision:
        f_precision.write(f"Epoch {epoch+1}: {precision:.4f}\n")
    with open("accuracy.txt", "a") as f_accuracy:
        f_accuracy.write(f"Epoch {epoch+1}: {accuracy:.4f}\n")

# Plot Losses
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.savefig("training_validation_loss_visdrone.png")
    plt.show()

# Plot Accuracy Map
def plot_accuracy_map(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.savefig("accuracy_map_visdrone.png")
    plt.show()

# Main Function
def fine_tune_clip():
    train_dir = r"D:/research proposall/Foggy Super Resolution/visdroneF/train"
    valid_dir = r"D:/research proposall/Foggy Super Resolution/visdroneF/valid"
    test_dir = r"D:/research proposall/Foggy Super Resolution/visdroneF/test"
    train_annotation_file = r"D:/research proposall/Foggy Super Resolution/visdroneF/train/annotations.jsonl"
    valid_annotation_file = r"D:/research proposall/Foggy Super Resolution/visdroneF/valid/annotations.jsonl"
    test_annotation_file = r"D:/research proposall/Foggy Super Resolution/visdroneF/test/annotations.jsonl"

    train_dataset = DetectionDataset(
        jsonl_file_path=train_annotation_file,
        image_directory_path=train_dir
    )
    val_dataset = DetectionDataset(
        jsonl_file_path=valid_annotation_file,
        image_directory_path=valid_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    lora_model = load_clip_model_with_lora()  # Add LoRA setup here
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=LR)

    lora_model.to(DEVICE)

    train_losses, val_losses, accuracies = train_model(lora_model, train_loader, val_loader, optimizer, DEVICE)
    lora_model.save_pretrained("FlorenceONVISDRONEMODEL")
    processor.save_pretrained("fine_tuned_florence_processorONVisdrone")

if __name__ == "__main__":
    fine_tune_clip()
