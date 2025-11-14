import lightning as L
import torch
from torchmetrics.functional import f1_score, precision, recall
from transformers import VideoMAEForVideoClassification

from src.utils.metrics import calculate_metrics


class VideoMAEModel(L.LightningModule):
    def __init__(self, learning_rate=1e-5, num_unfreeze_layers=1):
        super().__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-short-ssv2"
        )
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()

        for param in self.model.parameters():
            param.requires_grad = False

        if num_unfreeze_layers > 0:
            total_layers = len(self.model.videomae.encoder.layer)
            for i in range(1, num_unfreeze_layers + 1):
                for param in self.model.videomae.encoder.layer[
                    total_layers - i
                ].parameters():
                    param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.clip_outputs = []

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits
        return logits

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["pixel_values"], train_batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=x.size(0))
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=2)
        prec_score = precision(preds, y, task="multiclass", num_classes=2)
        recall_score = recall(preds, y, task="multiclass", num_classes=2)
        self.log("train_accuracy", acc, prog_bar=True, batch_size=x.size(0))
        self.log("train_f1", f1, prog_bar=True, batch_size=x.size(0))
        self.log("train_precision", prec_score, prog_bar=True, batch_size=x.size(0))
        self.log("train_recall", recall_score, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["pixel_values"], val_batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=x.size(0))
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=2)
        prec_score = precision(preds, y, task="multiclass", num_classes=2)
        recall_score = recall(preds, y, task="multiclass", num_classes=2)
        self.log("val_accuracy", acc, prog_bar=True, batch_size=x.size(0))
        self.log("val_f1", f1, prog_bar=True, batch_size=x.size(0))
        self.log("val_precision", prec_score, prog_bar=True, batch_size=x.size(0))
        self.log("val_recall", recall_score, prog_bar=True, batch_size=x.size(0))
        self.clip_outputs.append(
            {
                "preds": preds,
                "targets": y,
                "video_ids": val_batch["video_ids"],
                "start_times": val_batch["start_times"],
                "end_times": val_batch["end_times"],
                "video_labels": val_batch["video_labels"],
                "video_timestamps": val_batch["video_timestamps"],
            }
        )
        return loss

    def predict_step(self, test_batch, batch_idx):
        x, y = test_batch["pixel_values"], test_batch["labels"]
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_validation_epoch_end(self):
        metrics = calculate_metrics(self.clip_outputs)
        self.log("val_video_precision", metrics["precision"], prog_bar=True)
        self.log("val_video_recall", metrics["recall"], prog_bar=True)
        self.log("val_video_f1", metrics["f1"], prog_bar=True)
        self.clip_outputs = []
