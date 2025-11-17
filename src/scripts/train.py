import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from src.data.dataset import IWDDDataModule
from src.model.model import VideoClassificationModel
from src.utils.utils import get_model_config
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def main():
    L.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="videomae_ssv2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-layers", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--clip-duration", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--yolo-conf", type=float, default=0.3)
    parser.add_argument("--yolo-general-conf", type=float, default=0.5)
    args = parser.parse_args()

    model_config = get_model_config(args.model)
    print(f"Training model: {args.model}")
    print(f"Config: {model_config}")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", filename="best_model-{epoch:02d}-{val_loss:.2f}"
    )

    model = VideoClassificationModel(
        model_config=model_config,
        learning_rate=args.lr,
        num_unfreeze_layers=args.unfreeze_layers,
    )
    data = IWDDDataModule(
        model_config=model_config,
        batch_size=args.batch_size,
        num_workers=4,
        clip_duration=args.clip_duration,
        stride=args.stride,
        persistent_workers=True,
        train_split=0.7,
        num_frames=16,
        val_split=0.15,
        use_yolo=True,
        yolo_model_path=Path("runs/yolov8n_trash_detector/weights/best.pt"),
        yolo_general_model_path=Path("yolov8m.pt"),
        yolo_trash_conf=args.yolo_conf,
        yolo_general_conf=args.yolo_general_conf,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        log_every_n_steps=1,
        deterministic=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
