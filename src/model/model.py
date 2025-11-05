import lightning as L
import torch
from transformers import VideoMAEForVideoClassification


class VideoMAEModel(L.LightningModule):
    def __init__(self, learning_rate=1e-1):
        super().__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-short-ssv2"
        )
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits
        return logits

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_accuracy", acc, prog_bar=True)
        return loss

    def predict_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
