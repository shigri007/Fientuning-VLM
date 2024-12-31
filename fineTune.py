# %%
# @title Imports

import os
import re
import json
import torch
import base64
import numpy as np
import supervision as sv
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Generator
from peft import LoraConfig, get_peft_model
from PIL import Image
from roboflow import Roboflow
import matplotlib.pyplot as plt


# %%
#you can download the model direclty from hugging face,or use api
CHECKPOINT = "/home/ai/jupyter/MJ/hb/haider/haider/model"
REVISION = 'refs/pr/6'
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"

model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

# %%
# @title Example object detection inference



# %%
rf = Roboflow(api_key="")
project = rf.workspace("ram-q94bn").project("military-object2")
version = project.version(4)
dataset = version.download("florence2-od")

# %%
#!head -n 5 {dataset.location}/train/annotations.jsonl

# %%
# @title Define `DetectionsDataset` class

class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")


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

# %%
# @title Initiate `DetectionsDataset` and `DataLoader` for train and validation subsets

BATCH_SIZE = 6
NUM_WORKERS = 0

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers

train_dataset = DetectionDataset(
    jsonl_file_path = f"{dataset.location}/train/annotations.jsonl",
    image_directory_path = f"{dataset.location}/train/"
)
val_dataset = DetectionDataset(
    jsonl_file_path = f"{dataset.location}/valid/annotations.jsonl",
    image_directory_path = f"{dataset.location}/valid/"
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

# %%
# @title Setup LoRA Florence-2 model

config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
    revision=REVISION
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

# %%

# %%
# @title Run inference with pre-trained Florence-2 model on validation dataset

def render_inline(image: Image.Image, resize=(128, 128)):
    """Convert image into inline html."""
    image.resize(resize)
    with io.BytesIO() as buffer:
        image.save(buffer, format='jpeg')
        image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
        return f"data:image/jpeg;base64,{image_b64}"


def render_example(image: Image.Image, response):
    try:
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
        image = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image.copy(), detections)
        image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image, detections)
    except:
        print('failed to redner model response')
    return f"""
<div style="display: inline-flex; align-items: center; justify-content: center;">
    <img style="width:256px; height:256px;" src="{render_inline(image, resize=(128, 128))}" />
    <p style="width:512px; margin:10px; font-size:small;">{html.escape(json.dumps(response))}</p>
</div>
"""


def render_inference_results(model, dataset: DetectionDataset, count: int):
    html_out = ""
    count = min(count, len(dataset))
    for i in range(count):
        image, data = dataset.dataset[i]
        prefix = data['prefix']
        suffix = data['suffix']
        inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"], 
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        answer = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
        html_out += render_example(image, answer)


render_inference_results(peft_model, val_dataset, 4)

# %%
# @title Define train loop

# Directory setup for saving results
graph_dir = "/home/ai/jupyter/MJ/hb/haider/haider/graphres"
os.makedirs(graph_dir, exist_ok=True)

# Function to save training and validation loss plots
def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
# Set the device to GPU 2 (or fallback to CPU if not available)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def train_model(train_loader, val_loader, model, processor, epochs=1, lr=1e-6, output_dir='/home/ai/jupyter/MJ/hb/haider/haider/output_result', graph_dir='/home/ai/jupyter/MJ/hb/haider/haider/graphres'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)  # Ensure the graph directory exists
    # Move the model to the specified device
    model.to(DEVICE)
    
    # Use PyTorch AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=num_training_steps
    )

    # Track losses for each epoch
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):

            # Move inputs and labels to the correct device
            input_ids = inputs["input_ids"].to(DEVICE)
            pixel_values = inputs["pixel_values"].to(DEVICE)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                truncation=True, 
                max_length=1024,
                return_token_type_ids=False
            ).input_ids.to(DEVICE)

            # Forward pass
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}")

        # Save the train loss for graph
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{epochs}"):

                # Move inputs and labels to the correct device
                input_ids = inputs["input_ids"].to(DEVICE)
                pixel_values = inputs["pixel_values"].to(DEVICE)
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_token_type_ids=False
                ).input_ids.to(DEVICE)

                # Forward pass (no backpropagation for validation)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Validation Loss: {avg_val_loss}")

        # Save the validation loss for graph
        val_losses.append(avg_val_loss)

    # After training is done, plot and save the loss graph
        # After training is done, plot and save the loss graph
    plt.figure(figsize=(10, 6))
    epochs_range = [e + 1 for e in range(epochs)]
    plt.plot(epochs_range, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs_range, val_losses, label='Validation Loss', color='red')

    # Add data points on the curve
    plt.scatter(epochs_range, train_losses, color='blue', s=10, zorder=5)
    plt.scatter(epochs_range, val_losses, color='red', s=10, zorder=5)

    # Adjust x-axis ticks to display fewer labels (e.g., every 10th epoch)
    step_size = max(1, epochs // 25)  # Adjust tick frequency dynamically
    plt.xticks(ticks=epochs_range[::step_size])  # Show every 'step_size' epoch

    # Adjust y-axis ticks dynamically
    max_loss = max(max(train_losses), max(val_losses))
    plt.yticks(np.arange(0.5, max_loss + 1, 0.5))

    # Labels, title, legend, and grid
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    
    # Save the plot to the specified directory
    loss_graph_path = os.path.join(graph_dir, 'training_and_validation_loss.png')
    plt.savefig(loss_graph_path)
    plt.close()

    print(f"Training and validation loss graph saved at {loss_graph_path}")

# Extract CLASSES
def extract_classes(dataset):
    class_set = set()
    for i in range(len(dataset.dataset)):
        _, data = dataset.dataset[i]
        suffix = data["suffix"]
        # Replace the regular expression pattern with your specific pattern for class extraction
        classes = re.findall(r'([a-zA-Z0-9 ]+ of [a-zA-Z0-9 ]+)<loc_\d+>', suffix)
        class_set.update(classes)
    return sorted(class_set)

def collect_predictions_and_targets(model, processor, dataset, classes):
    targets = []
    predictions = []

    for i in range(len(dataset.dataset)):
        image, data = dataset.dataset[i]
        prefix = data['prefix']
        suffix = data['suffix']

        # Prepare input
        inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)

        # Generate predictions
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Post-process predictions
        prediction = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)

        # Check if predictions are empty using len()
        if len(prediction) == 0:
            print(f"Warning: No predictions for image {i}, skipping.")
            continue  # Skip this sample if no predictions are made

        # Handle missing classes in predictions
        prediction = prediction[np.isin(prediction['class_name'], classes)]
        prediction.class_id = np.array([
            classes.index(class_name) if class_name in classes else -1
            for class_name in prediction['class_name']
        ])
        prediction.confidence = np.ones(len(prediction))

        # Post-process targets
        target = processor.post_process_generation(suffix, task='<OD>', image_size=image.size)
        target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)

        # Handle missing classes in targets
        target.class_id = np.array([
            classes.index(class_name) if class_name in classes else -1
            for class_name in target['class_name']
        ])

        # Check if targets are empty using len()
        if len(target) == 0:
            print(f"Warning: No targets for image {i}, skipping.")
            continue  # Skip this sample if no targets are found

        # Append results if valid predictions and targets exist
        targets.append(target)
        predictions.append(prediction)

    return predictions, targets

    fig, ax = plt.subplots(figsize=(8, 6))
    confusion_matrix.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix plot saved at {file_path}")

# Run training
EPOCHS = 250
LR = 5e-6

train_model(train_loader, val_loader, peft_model, processor, epochs=EPOCHS, lr=LR, graph_dir='/home/ai/jupyter/MJ/hb/haider/haider/graphres')

# Extract CLASSES
CLASSES = extract_classes(train_dataset)

# Collect predictions and targets for validation dataset
predictions, targets = collect_predictions_and_targets(peft_model, processor, val_dataset, CLASSES)

# Calculate and save metrics
print(f"All results saved in {graph_dir}")
