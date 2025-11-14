import lightning as L
import torch
from torchmetrics.functional import f1_score, precision, recall

from src.utils.metrics import calculate_metrics


class VideoClassificationModel(L.LightningModule):
    def __init__(
        self,
        model_config,
        learning_rate=1e-5,
        num_unfreeze_layers=1,
        num_classes=2,
    ):
        super().__init__()
        self.config = model_config
        self.num_classes = num_classes
        self.num_unfreeze_layers = num_unfreeze_layers
        self.model = self.load_model()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.setup_unfreezing()
        self.clip_outputs = []

    def load_model(self):
        model_class_name = self.config["model_class"]
        model_name = self.config["model_name"]

        if "XCLIP" in model_class_name:
            from transformers import XCLIPVisionModel

            model = XCLIPVisionModel.from_pretrained(model_name)
            model.classifier = torch.nn.Linear(
                model.config.projection_dim, self.num_classes
            )
            return model
        else:
            from transformers import VideoMAEForVideoClassification

            model = VideoMAEForVideoClassification.from_pretrained(
                model_name, num_labels=self.num_classes, ignore_mismatched_sizes=True
            )
        return model

    def setup_unfreezing(self):
        for param in self.model.parameters():
            param.requires_grad = False

        if self.num_unfreeze_layers > 0:
            if "VideoMAE" in self.config["model_class"]:
                total_layers = len(self.model.videomae.encoder.layer)
                layers = self.model.videomae.encoder.layer
            else:
                total_layers = len(self.model.vision_model.encoder.layers)
                layers = self.model.vision_model.encoder.layers
            for i in range(1, self.num_unfreeze_layers + 1):
                for param in layers[total_layers - i].parameters():
                    param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        if "XCLIP" in self.config["model_class"]:
            pass
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
