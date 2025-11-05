import lightning as L


class YoloModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # self.model =

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
