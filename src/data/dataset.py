import lightning as L
from torch.utils.data import Dataset


class YoloDataModule(L.LightningDataModule):
    pass


class IWDDDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class IWDDDataModule(L.LightningDataModule):
    pass
