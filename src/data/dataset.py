import json
from pathlib import Path

import lightning as L
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import VideoMAEImageProcessor

# from torchvision import transforms


class VideoFolder(Dataset):
    def __init__(
        self, videos_dir, labels_dir, stride=1, clip_duration=3, num_frames=16
    ):
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
        self.clip_duration = clip_duration
        self.clips = []
        self.stride = stride
        self.num_frames = num_frames
        self.prepare_clips()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]

        encoded_video = EncodedVideo.from_path(clip_info["video_path"])

        video_data = encoded_video.get_clip(
            start_sec=clip_info["start_time"], end_sec=clip_info["end_time"]
        )
        frames = video_data["video"]

        total_frames = frames.shape[1]

        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = frames[:, indices, :, :]

        permutated = frames.permute(1, 0, 2, 3)
        frame_list = [frame for frame in permutated]

        inputs = self.processor(frame_list, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(clip_info["label"], dtype=torch.long),
            "video_id": clip_info["video_id"],
            "video_label": clip_info["video_label"],
            "start_time": clip_info["start_time"],
            "end_time": clip_info["end_time"],
            "video_timestamp": clip_info["video_timestamp"],
        }

    def prepare_clips(self):
        for video_path, label_path in self.samples:
            with open(label_path) as f:
                annotation = json.load(f)
                label = annotation["Dumping"]
            if label == 1:
                timestamp = annotation["DumpingDetails"]["Timestamp"]
            else:
                timestamp = -1
                # type_of_dumping = annotation["DumpingDetails"]["Type of Dumping"]

            start_time = 0
            encoded_video = EncodedVideo.from_path(video_path)
            duration = encoded_video.duration
            while start_time < duration:
                end_time = start_time + self.clip_duration
                if end_time > duration:
                    end_time = duration
                    start_time = end_time - self.clip_duration

                clip_label = 0
                if label == 1:
                    if start_time <= timestamp < end_time:
                        clip_label = 1

                self.clips.append(
                    {
                        "video_path": video_path,
                        "label": clip_label,
                        "start_time": start_time,
                        "end_time": end_time,
                        "video_id": video_path.stem,
                        "video_label": label,
                        "video_timestamp": timestamp,
                    }
                )
                start_time += self.stride


class IWDDDataModule(L.LightningDataModule):
    def __init__(
        self,
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
    ):
        super().__init__()
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.train_split = train_split
        self.val_split = val_split
        self.clip_duration = clip_duration
        self.stride = stride
        self.num_frames = num_frames
    
    def collate_fn(self, batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["label"] for item in batch]
        video_ids = [item["video_id"] for item in batch]
        start_times = [item["start_time"] for item in batch]
        end_times = [item["end_time"] for item in batch]
        video_labels = [item["video_label"] for item in batch]
        video_timestamps = [item["video_timestamp"] for item in batch]
        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.stack(labels),
            "video_ids": video_ids,
            "start_times": start_times,
            "end_times": end_times,
            "video_labels": video_labels,
            "video_timestamps": video_timestamps,
        }

    def setup(self, stage: str):

        full_dataset = VideoFolder(
            self.videos_dir,
            self.annotations_dir,
            self.stride,
            self.clip_duration,
            self.num_frames,
        )
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

        # self.train_transform = transforms.Compose([])
        # self.val_transform = transforms.Compose([])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
        )
