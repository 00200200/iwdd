import lightning as L
import torch
from transformers import VideoMAEForVideoClassification
from torchmetrics.functional import f1_score, precision, recall


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
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=2)
        precision = precision(preds, y, task="multiclass", num_classes=2)
        recall = recall(preds, y, task="multiclass", num_classes=2)
        self.log("train_accuracy", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        self.log("train_precision", precision, prog_bar=True)
        self.log("train_recall", recall, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=2)
        precision = precision(preds, y, task="multiclass", num_classes=2)
        recall = recall(preds, y, task="multiclass", num_classes=2)
        self.log("val_accuracy", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        return loss

    def predict_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
