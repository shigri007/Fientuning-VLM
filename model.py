import os
import torch
from transformers import AutoProcessor
from tqdm import tqdm
from utils import save_loss_plot


def train_model(train_loader, val_loader, model, processor, epochs, lr, graph_dir, save_dir):
    """
    Trains the model, validates it after each epoch, and saves the model state.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = inputs["input_ids"].to(model.device)
            pixel_values = inputs["pixel_values"].to(model.device)
            labels = processor.tokenizer(
                text=answers, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(model.device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}: Training Loss = {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{epochs}"):
                input_ids = inputs["input_ids"].to(model.device)
                pixel_values = inputs["pixel_values"].to(model.device)
                labels = processor.tokenizer(
                    text=answers, return_tensors="pt", padding=True, truncation=True
                ).input_ids.to(model.device)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss:.4f}")

        # Save the model after each epoch
       # model_save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pt")
       # torch.save(model.state_dict(), model_save_path)
       # print(f"Model saved to {model_save_path}")

    # Save the loss plot
    save_loss_plot(train_losses, val_losses, graph_dir)
