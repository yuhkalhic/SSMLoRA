from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingConfig:
    # Model configurations
    model_path: str = "/datas/huggingface/Llama-2-7b-hf"
    num_labels: int = 2
    
    # Training hyperparameters
    learning_rate: float = 5e-6
    batch_size: int = 8
    max_epochs: int = 30
    num_workers: int = 4
    max_length: int = 256
    gradient_clip_val: float = 1.0
    
    # LoRA configurations
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Which components to apply LoRA
    lora_query: bool = True
    lora_key: bool = True
    lora_value: bool = True
    lora_projection: bool = True
    lora_mlp: bool = True
    lora_head: bool = True
    
    # Hardware and logging
    device: int = 0
    precision: str = "16-mixed"
    enable_progress_bar: bool = True
    log_every_n_steps: int = 10
    
    # Early stopping
    early_stopping_patience: int = 4
    scheduler_patience: int = 2
    scheduler_factor: float = 0.1

    # Paths
    data_base_path: str = "/datas/yujiayang/combined"
    save_dir: str = "logs/"

@dataclass
class DatasetConfig:
    name: str  # e.g., "WSC", "WiC"
    dataset_class: str  # e.g., "WSCDataset", "WiCDataset"


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset specific configuration"""
    configs = {
        "WSC": DatasetConfig(
            name="WSC",
            dataset_class="WSCDataset"
        ),
        "WiC": DatasetConfig(
            name="WiC",
            dataset_class="WiCDataset"
        ),
        "MultiRC": DatasetConfig(
            name="MultiRC",
            dataset_class="MultiRCDataset"
        ),
        "COPA": DatasetConfig(
            name="COPA",
            dataset_class="COPADataset"
        ),
        "BoolQ": DatasetConfig(
            name="BoolQ",
            dataset_class="BoolQDataset"
        ),
    }
    return configs.get(dataset_name)