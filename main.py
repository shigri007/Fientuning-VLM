from train import train_model
from utils import save_loss_plot, extract_classes
from evaluate import collect_predictions_and_targets, render_inference_results
from config import (
    DEVICE,
    EPOCHS,
    LR,
    train_loader,
    val_loader,
    train_dataset,
    val_dataset,
    processor,
    peft_model,
    graph_dir,
)

if __name__ == "__main__":
    # Train the model
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=peft_model,
        processor=processor,
        epochs=EPOCHS,
        lr=LR,
        graph_dir=graph_dir,
    )

    # Extract classes
    CLASSES = extract_classes(train_dataset)

    # Collect predictions and targets
    predictions, targets = collect_predictions_and_targets(
        model=peft_model,
        processor=processor,
        dataset=val_dataset,
        classes=CLASSES,
    )

    # Render inference results
    render_inference_results(peft_model, val_dataset, count=4)

    print(f"All results saved in {graph_dir}")
