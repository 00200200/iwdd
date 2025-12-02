import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from transformers import AutoProcessor
from ultralytics import YOLO


class VideoFolder(Dataset):
    def __init__(
        self,
        videos_dir,
        labels_dir,
        model_config,
        stride=1,
        clip_duration=3,
        num_frames=16,
        use_yolo=False,
        yolo_model_path=None,
        yolo_general_model_path=None,
        yolo_trash_conf=0.2,
        yolo_general_conf=0.5,
        use_augmentation=False,
    ):
        self.videos_dir = Path(videos_dir)
        self.labels_dir = Path(labels_dir)
        self.model_config = model_config
        self.use_text = self.model_config["use_text"]
        self.text_prompts = self.model_config["text_prompts"] if self.use_text else None
        self.processor = self.load_processor()

        self.use_yolo = use_yolo
        self.yolo_trash_conf = yolo_trash_conf
        self.yolo_general_conf = yolo_general_conf
        self.use_augmentation = use_augmentation

        if use_yolo and yolo_model_path:
            self.yolo_model = YOLO(yolo_model_path)
        else:
            self.yolo_model = None

        if use_yolo and yolo_general_model_path:
            self.yolo_general_model = YOLO(yolo_general_model_path)
        else:
            self.yolo_general_model = None

        if self.use_augmentation:
            self.augmentation = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.5, hue=0.3),
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
                ]
            )
        else:
            self.augmentation = None

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

    def load_processor(self):
        model_name = self.model_config["model_name"]
        processor = AutoProcessor.from_pretrained(model_name)
        image_processor = getattr(processor, "image_processor", processor)
        if hasattr(image_processor, "do_rescale"):
            image_processor.do_rescale = False
        return processor

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]

        encoded_video = EncodedVideo.from_path(clip_info["video_path"])
        video_data = encoded_video.get_clip(
          start_sec=clip_info["start_time"], 
          end_sec=clip_info["end_time"]
        )
        frames = video_data["video"]
        frames = frames / 255.0

        total_frames = frames.shape[1]
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = frames[:, indices, :, :]
        if self.use_yolo and self.yolo_model:
            frames = self._apply_bounding_box(frames)

        if self.augmentation:
            frames = self.apply_augmentation(frames)

        permutated = frames.permute(1, 0, 2, 3)


        if self.use_text == 1:
            # 2 prompts -> two classes
            inputs = self.processor(
                text=self.text_prompts, 
                images=list(permutated),
                return_tensors="pt",
                padding=True
                )
        else:
            frame_list = [frame for frame in permutated]
            inputs = self.processor(images=frame_list, return_tensors="pt")
        
        pixel_values = inputs["pixel_values"].squeeze(0)

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        if input_ids is not None:
            input_ids = input_ids.squeeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(clip_info["label"], dtype=torch.long),
            "video_id": clip_info["video_id"],
            "video_label": clip_info["video_label"],
            "start_time": clip_info["start_time"],
            "end_time": clip_info["end_time"],
            "video_timestamp": clip_info["video_timestamp"]
            }

    def apply_augmentation(self, frames):
        batch_size, num_frames, height, width = frames.shape

        augmented_frames = frames.clone()

        for frame_idx in range(num_frames):
            frame = frames[:, frame_idx, :, :]
            frame = self.augmentation(frame)
            frame = torch.clamp(frame, 0.0, 1.0)
            augmented_frames[:, frame_idx, :, :] = frame

        return augmented_frames

    def _apply_bounding_box(self, frames):
        _, num_frames, _, _ = frames.shape

        frames_with_boxes = frames.clone()

        # self.yolo_general_model: person=0, car=2, truck=7
        general_class_ids = [0, 2, 7]

        for frame_idx in range(num_frames):
            frame = (
                (frames[:, frame_idx, :, :] * 255.0)
                .permute(1, 2, 0)
                .numpy()
                .astype(np.uint8)
            )

            annotated_frame = frame.copy()

            if self.yolo_model:
                trash_result = self.yolo_model(
                    frame,
                    conf=self.yolo_trash_conf,
                    iou=0.3
                )[0]
                annotated_frame = trash_result.plot(img=annotated_frame, labels=False)

            if self.yolo_general_model:
                general_result = self.yolo_general_model(
                    frame,
                    conf=self.yolo_general_conf,
                    classes=general_class_ids
                )[0]
                general_result.boxes = general_result.boxes[
                    [int(c.item()) in general_class_ids for c in general_result.boxes.cls]
                ]

                annotated_frame = general_result.plot(img=annotated_frame, labels=False)

            frames_with_boxes[:, frame_idx, :, :] = (
                torch.from_numpy(annotated_frame.transpose(2, 0, 1)).float() / 255.0
            )

        return frames_with_boxes

    def prepare_clips(self):
        for video_path, label_path in self.samples:
            with open(label_path) as f:
                annotation = json.load(f)
                label = annotation["Dumping"]
            if label == 1:
                timestamp = annotation["DumpingDetails"]["Timestamp"]
            else:
                timestamp = -1

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
                if end_time >= duration:
                    break
                start_time += self.stride


class IWDDDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_config,
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
        yolo_model_path=None,
        yolo_general_model_path=None,
        use_yolo=False,
        yolo_trash_conf=0.3,
        yolo_general_conf=0.5,
        use_augmentation=False,
    ):
        super().__init__()
        self.model_config = model_config
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
        self.use_yolo = use_yolo
        self.yolo_model_path = yolo_model_path
        self.yolo_general_model_path = yolo_general_model_path
        self.yolo_trash_conf = yolo_trash_conf
        self.yolo_general_conf = yolo_general_conf
        self.use_augmentation = use_augmentation

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
    
    def collate_fn_wtext(self, batch):
        pixel_values = [item["pixel_values"] for item in batch]
        input_ids = [item.get("input_ids") for item in batch]
        attention_mask = [item.get("attention_mask") for item in batch]
        labels = [item["label"] for item in batch]
        video_ids = [item["video_id"] for item in batch]
        start_times = [item["start_time"] for item in batch]
        end_times = [item["end_time"] for item in batch]
        video_labels = [item["video_label"] for item in batch]
        video_timestamps = [item["video_timestamp"] for item in batch]
        return {
            "pixel_values": torch.stack(pixel_values),
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "video_ids": video_ids,
            "start_times": start_times,
            "end_times": end_times,
            "video_labels": video_labels,
            "video_timestamps": video_timestamps,
        }

    def setup(self, stage: str):

        temp_dataset = VideoFolder(
            self.videos_dir,
            self.annotations_dir,
            self.model_config,
            self.stride,
            self.clip_duration,
            self.num_frames,
            self.use_yolo,
            self.yolo_model_path,
            self.yolo_general_model_path,
            self.yolo_trash_conf,
            self.yolo_general_conf,
            use_augmentation=False,
        )
        total = len(temp_dataset)
        train_size = int(total * self.train_split)
        val_size = int(total * self.val_split)
        test_size = total - train_size - val_size
        train_subset, val_subset, test_subset = random_split(
            temp_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_dataset_full = VideoFolder(
            self.videos_dir,
            self.annotations_dir,
            self.model_config,
            self.stride,
            self.clip_duration,
            self.num_frames,
            self.use_yolo,
            self.yolo_model_path,
            self.yolo_general_model_path,
            self.yolo_trash_conf,
            self.yolo_general_conf,
            use_augmentation=self.use_augmentation,
        )
        self.val_dataset_full = VideoFolder(
            self.videos_dir,
            self.annotations_dir,
            self.model_config,
            self.stride,
            self.clip_duration,
            self.num_frames,
            self.use_yolo,
            self.yolo_model_path,
            self.yolo_general_model_path,
            self.yolo_trash_conf,
            self.yolo_general_conf,
            use_augmentation=False,
        )
        self.train_dataset = Subset(self.train_dataset_full, train_subset.indices)
        self.val_dataset = Subset(self.val_dataset_full, val_subset.indices)
        self.test_dataset = Subset(self.val_dataset_full, test_subset.indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn_wtext if self.model_config["use_text"] == 1 else self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn_wtext if self.model_config["use_text"] == 1 else self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn_wtext if self.model_config["use_text"] == 1 else self.collate_fn,
        )
