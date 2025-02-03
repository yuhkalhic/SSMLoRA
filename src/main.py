import argparse
import torch
import importlib
from config.default_config import TrainingConfig, get_dataset_config
from data.data_utils import load_dataset, setup_tokenizer, get_dataloaders
from models.model_utils import prepare_model, count_parameters
from models.ssm_lora import TimeAxis
from training.lightning_module import CustomLightningModule
from training.train_utils import setup_trainer, run_training

def parse_args():
    parser = argparse.ArgumentParser(description='SSM-LoRA Training Script')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (WSC/WiC)')
    
    config = TrainingConfig()
    for key, value in vars(config).items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    return parser.parse_args()

def main():
    args = parse_args()
    config = TrainingConfig(**{k: v for k, v in vars(args).items() if k in vars(TrainingConfig())})
    
    dataset_config = get_dataset_config(args.dataset)
    if dataset_config is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataset_module = importlib.import_module(f"data.{dataset_config.name.lower()}_dataset")
    dataset_class = getattr(dataset_module, dataset_config.dataset_class)
    
    print("Setting up training components...")
    tokenizer = setup_tokenizer(config.model_path)
    dataset_dict = load_dataset(dataset_config.name, config.data_base_path)
    train_loader, val_loader = get_dataloaders(
        dataset_class,
        dataset_dict,
        tokenizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    print("Initializing model...")
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    time_axis = TimeAxis(device=device)
    model = prepare_model(config, time_axis)
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    trainable_params = count_parameters(model)
    print(f"Total trainable parameters: {trainable_params:,}")
    
    print("Setting up training...")
    lightning_model = CustomLightningModule(model, config)
    trainer = setup_trainer(config, dataset_config.name)
    
    print(f"Starting training on {dataset_config.name}...")
    results = run_training(trainer, lightning_model, train_loader, val_loader)
    
    print("\nTraining completed!")
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    print(f"Training time: {results['training_time'] / 60:.2f} minutes")

if __name__ == "__main__":
    main()