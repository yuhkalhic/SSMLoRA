import os
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def load_dataset(dataset_name, base_path):
    """Load dataset from jsonl files"""
    dataset_path = os.path.join(base_path, dataset_name)
    data = {}
    for split in ["train", "val"]:
        split_path = os.path.join(dataset_path, f"{split}.jsonl")
        with open(split_path, "r") as f:
            data[split] = [json.loads(line) for line in f]
    return data

def setup_tokenizer(model_path):
    """Setup tokenizer with proper configurations"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return tokenizer

def get_dataloaders(dataset_cls, dataset_dict, tokenizer, batch_size=8, num_workers=4):
    """Create train and validation dataloaders"""
    train_dataset = dataset_cls(dataset_dict["train"], tokenizer=tokenizer)
    val_dataset = dataset_cls(dataset_dict["val"], tokenizer=tokenizer)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return train_loader, val_loader