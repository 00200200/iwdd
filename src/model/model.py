import lightning as L
import torch
import torch.nn as nn
from torchmetrics.functional import f1_score, precision, recall
from transformers import AutoModel

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
        model_name = self.config["model_name"]
        model = AutoModel.from_pretrained(model_name)
        if hasattr(model.config, "hidden_size"):
            output_dim = model.config.hidden_size
        elif hasattr(model.config, "projection_dim"):
            output_dim = model.config.projection_dim
        model.classifier = nn.Linear(output_dim, self.num_classes)
        return model

    def setup_unfreezing(self):
        for param in self.model.parameters():
            param.requires_grad = False

        if self.num_unfreeze_layers > 0:
            if "videomae" in self.config["model_name"]:
                total_layers = len(self.model.encoder.layer)
                layers = self.model.encoder.layer
            else:
                total_layers = len(self.model.vision_model.encoder.layers)
                layers = self.model.vision_model.encoder.layers
            for i in range(1, self.num_unfreeze_layers + 1):
                for param in layers[total_layers - i].parameters():
                    param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        pixel_values = inputs["pixel_values"]
        if self.config["use_text"] == 1:
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Text input is required for this model.")
            attention_mask = inputs["attention_mask"]
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids[0],
                attention_mask=attention_mask[0],
            )
            return outputs.logits_per_video
        outputs = self.model(pixel_values)
        sequence_output = outputs.last_hidden_state[:, 0]
        logits = self.model.classifier(sequence_output)
        return logits

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch, train_batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=x["pixel_values"].size(0))
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=2)
        prec_score = precision(preds, y, task="multiclass", num_classes=2)
        recall_score = recall(preds, y, task="multiclass", num_classes=2)
        self.log("train_accuracy", acc, prog_bar=True, batch_size=x["pixel_values"].size(0))
        self.log("train_f1", f1, prog_bar=True, batch_size=x["pixel_values"].size(0))
        self.log("train_precision", prec_score, prog_bar=True, batch_size=x["pixel_values"].size(0))
        self.log("train_recall", recall_score, prog_bar=True, batch_size=x["pixel_values"].size(0))
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch, val_batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=x["pixel_values"].size(0))
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = f1_score(preds, y, task="multiclass", num_classes=2)
        prec_score = precision(preds, y, task="multiclass", num_classes=2)
        recall_score = recall(preds, y, task="multiclass", num_classes=2)
        self.log("val_accuracy", acc, prog_bar=True, batch_size=x["pixel_values"].size(0))
        self.log("val_f1", f1, prog_bar=True, batch_size=x["pixel_values"].size(0))
        self.log("val_precision", prec_score, prog_bar=True, batch_size=x["pixel_values"].size(0))
        self.log("val_recall", recall_score, prog_bar=True, batch_size=x["pixel_values"].size(0))
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
        x, y = test_batch, test_batch["labels"]
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
