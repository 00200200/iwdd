import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from src.data.dataset import IWDDDataModule
from src.model.model import VideoMAEModel


def main():
    L.seed_everything(42)

    model = VideoMAEModel(learning_rate=1e-5, num_unfreeze_layers=1)
    data = IWDDDataModule(
        videos_dir="data/raw/videos",
        annotations_dir="data/raw/labels",
        batch_size=8,
        num_workers=4,
        clip_duration=3,
        stride=1,
        persistent_workers=True,
        train_split=0.7,
        num_frames=16,
        val_split=0.15,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", filename="best_model-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="cuda",
        log_every_n_steps=1,
        deterministic=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
