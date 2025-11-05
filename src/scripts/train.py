import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

# from src.data.dataio import ImageDataModule
# from src.model.model import Resnet50


def main():
    L.seed_everything(42)

    # model = Resnet50()
    # data = ImageDataModule()

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss", filename="best_model-{epoch:02d}-{val_loss:.2f}"
    # )

    # trainer = L.Trainer(
    #     max_epochs=20,
    #     accelerator="auto",
    #     log_every_n_steps=1,
    #     deterministic=True,
    #     callbacks=[checkpoint_callback],
    # )
    # trainer.fit(model, data)


if __name__ == "__main__":
    main()
