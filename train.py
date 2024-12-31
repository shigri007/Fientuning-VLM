import os
import torch
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
from config import DEVICE


def train_model(train_loader, val_loader, model, processor, epochs, lr, graph_dir):
    os.makedirs(graph_dir, exist_ok=True)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_training_steps)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = inputs["input_ids"].to(DEVICE)
            pixel_values = inputs["pixel_values"].to(DEVICE)
            labels = processor.tokenizer(
                text=answers, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).input_ids.to(DEVICE)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{epochs}"):
                input_ids = inputs["input_ids"].to(DEVICE)
                pixel_values = inputs["pixel_values"].to(DEVICE)
                labels = processor.tokenizer(
                    text=answers, return_tensors="pt", padding=True, truncation=True, max_length=1024
                ).input_ids.to(DEVICE)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    # Plot and save the loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", color="blue")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graph_dir, "training_and_validation_loss.png"))
    plt.close()
