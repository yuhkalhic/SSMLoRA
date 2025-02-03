import pytorch_lightning as L
import torch
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CustomLightningModule(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                      labels=batch["label"])
        self.log("train_loss", outputs["loss"])
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                      labels=batch["label"])
        self.log("val_loss", outputs["loss"], prog_bar=True)

        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.val_accuracy(predicted_labels, batch["label"])
        self.log("val_accuracy", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                      labels=batch["label"])
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.test_accuracy(predicted_labels, batch["label"])
        self.log("test_accuracy", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode="max",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "frequency": 1
            },
        }