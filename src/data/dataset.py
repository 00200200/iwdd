import json
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import VideoMAEImageProcessor  # VideoMaeVideoProcessor > <


class VideoFolder(Dataset):
    def __init__(self, videos_dir, labels_dir):
        self.videos_dir = Path(videos_dir)
        self.labels_dir = Path(labels_dir)
        self.processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-short-ssv2"
        )

        video_files = sorted(self.videos_dir.glob("*.mp4"))

        self.samples = []
        for video_path in video_files:
            label_path = self.labels_dir / f"{video_path.stem}.json"
            if label_path.exists():
                self.samples.append((video_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label_path = self.samples[idx]

        # inputs = self.processor(str(video_path), return_tensors="pt")
        # pixel_values = inputs["pixel_values"].squeeze(0)

        pass
        # with open(label_path) as f:
        #     label = json.load(f)["Dumping"]

        # return pixel_values, torch.tensor(label, dtype=torch.long)


class IWDDDataModule(L.LightningDataModule):
    def __init__(
        self,
        videos_dir="data/raw/videos",
        annotations_dir="data/raw/labels",
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        train_split=0.7,
        val_split=0.15,
    ):
        super().__init__()
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.train_split = train_split
        self.val_split = val_split

    def setup(self, stage: str):

        full_dataset = VideoFolder(self.videos_dir, self.annotations_dir)
        total = len(full_dataset)
        train_size = int(total * self.train_split)
        val_size = int(total * self.val_split)
        test_size = total - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        # if stage == "fit":
        #     self.train_dataset = VideoFolder(self.videos_dir, self.annotations_dir)
        #     self.val_dataset = VideoFolder(self.videos_dir, self.annotations_dir)
        # else:
        #     self.test_dataset = VideoFolder(self.videos_dir, self.annotations_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
