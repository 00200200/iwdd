import matplotlib.pyplot as plt

from src.data.dataset import VideoFolder

model_config = {"model_name": "MCG-NJU/videomae-base"}
yolo_model_path = "models/best.pt"
yolo_general_model_path = "models/yolo11m.pt"
videos_dir = "data/raw/videos"
labels_dir = "data/raw/labels"
clip_indices = list(range(1, 100, 18))


datasets = {
    "Baseline (no YOLO, no Aug)": VideoFolder(
        model_config=model_config,
        use_yolo=False,
        use_augmentation=False,
        videos_dir=videos_dir,
        labels_dir=labels_dir,
    ),
    "YOLO only": VideoFolder(
        model_config=model_config,
        use_yolo=True,
        use_augmentation=False,
        yolo_model_path=yolo_model_path,
        yolo_general_model_path=yolo_general_model_path,
        videos_dir=videos_dir,
        labels_dir=labels_dir,
    ),
    "Augmentation only": VideoFolder(
        model_config=model_config,
        use_yolo=False,
        use_augmentation=True,
        videos_dir=videos_dir,
        labels_dir=labels_dir,
    ),
    "YOLO + Augmentation": VideoFolder(
        model_config=model_config,
        use_yolo=True,
        use_augmentation=True,
        yolo_model_path=yolo_model_path,
        yolo_general_model_path=yolo_general_model_path,
        videos_dir=videos_dir,
        labels_dir=labels_dir,
    ),
}


def visualize_clip(clip_idx: int) -> None:
    print(f"\n=== CLIP {clip_idx} ===")
    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, dataset) in zip(axes, datasets.items()):
        sample = dataset[clip_idx]
        middle_frame_idx = sample["pixel_values"].shape[0] // 2
        frame = sample["pixel_values"][middle_frame_idx]

        frame_np = frame.permute(1, 2, 0).numpy()
        frame_display = frame_np
        if frame_display.min() < 0 or frame_display.max() > 1:
            frame_display = (frame_display - frame_display.min()) / (
                frame_display.max() - frame_display.min()
            )

        ax.imshow(frame_display)
        ax.set_title(f"{name}\nLabel: {sample['label'].item()}")
        ax.axis("off")

    plt.suptitle(f"Clip {clip_idx} - Video: {sample['video_id']}")
    plt.tight_layout()
    plt.savefig(f"results/clip_comparison_{clip_idx}.png")


if __name__ == "__main__":
    for clip_idx in clip_indices:
        visualize_clip(clip_idx)
