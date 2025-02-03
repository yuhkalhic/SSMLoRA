import os
import time
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

def setup_trainer(config, dataset_name):
    """Setup training environment with callbacks and logger"""
    callbacks = [
        ModelCheckpoint(
            save_top_k=1,
            mode="max",
            monitor="val_accuracy"
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.early_stopping_patience,
            verbose=True
        )
    ]

    logger = CSVLogger(
        save_dir=config.save_dir,
        name=f"{dataset_name.lower()}-model-r{config.lora_r}"
    )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=config.gradient_clip_val,
        accelerator="gpu",
        precision=config.precision,
        devices=[config.device],
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        enable_progress_bar=config.enable_progress_bar
    )

    return trainer

def run_training(trainer, model, train_loader, val_loader):
    """Execute training process and return results"""
    start = time.time()
    
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    elapsed = time.time() - start
    
    # Test on both train and validation sets
    train_results = trainer.test(model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_results = trainer.test(model, dataloaders=val_loader, ckpt_path="best", verbose=False)
    
    return {
        "train_accuracy": train_results[0]['test_accuracy'],
        "val_accuracy": val_results[0]['test_accuracy'],
        "training_time": elapsed
    }