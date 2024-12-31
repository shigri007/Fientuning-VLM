import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
from dataset import DetectionDataset

# Paths and device setup
CHECKPOINT = "/home/ai/jupyter/MJ/hb/haider/haider/model"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
graph_dir = "/home/ai/jupyter/MJ/hb/haider/haider/graphres"

# Model and processor initialization
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

# LoRA configuration
config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
peft_model = get_peft_model(model, config)

# Dataset setup
train_dataset = DetectionDataset(jsonl_file_path="path/to/train.jsonl", image_directory_path="path/to/train")
val_dataset = DetectionDataset(jsonl_file_path="path/to/val.jsonl", image_directory_path="path/to/val")
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6)
EPOCHS = 250
LR = 5e-6
