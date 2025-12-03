import torch
from pytorchvideo.data.encoded_video import EncodedVideo

from src.model.model import VideoClassificationModel
from src.utils.utils import get_model_config


def load_model(model_name, path):
    model_config = get_model_config(model_name)
    model = VideoClassificationModel.load_from_checkpoint(
        path, model_config=model_config
    )
    model.eval()
    return model


def load_processor(model_name):
    model_config = get_model_config(model_name)
    processor_class = model_config["processor_class"]
    model_name_hf = model_config["model_name"]

    if "VideoMAE" in processor_class:
        from transformers import VideoMAEImageProcessor

        return VideoMAEImageProcessor.from_pretrained(model_name_hf)
    elif "XCLIP" in processor_class:
        from transformers import XCLIPProcessor

        return XCLIPProcessor.from_pretrained(model_name_hf)


def extract_clips(video_path, clip_duration=3, stride=1, num_frames=16):
    encoded_video = EncodedVideo.from_path(str(video_path))
    duration = encoded_video.duration

    clips = []
    start_time = 0.0

    while start_time < duration:
        end_time = start_time + clip_duration
        if end_time > duration:
            end_time = duration
            start_time = end_time - clip_duration

        video_data = encoded_video.get_clip(start_sec=start_time, end_sec=end_time)
        encoded_video.close()
        frames = video_data["video"]

        total_frames = frames.shape[1]

        indices = torch.linspace(0, total_frames - 1, num_frames).long()
        frames = frames[:, indices, :, :]

        frames = frames.permute(1, 0, 2, 3)

        clips.append({"frames": frames, "start_time": start_time, "end_time": end_time})

        start_time += stride
        if start_time >= duration:
            break
    

    return clips


def process_clip(frames, processor):
    frame_list = [frame for frame in frames]
    inputs = processor(frame_list, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    return pixel_values


def predict_video(model, processor, video_path):
    clips = extract_clips(video_path)
    clip_predictions = []
    with torch.no_grad():
        for clip in clips:
            pixel_values = process_clip(clip["frames"], processor)
            pixel_values = pixel_values
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()

            clip_predictions.append(
                {
                    "prediction": pred,
                    "confidence": confidence,
                    "start_time": clip["start_time"],
                    "end_time": clip["end_time"],
                }
            )

    positive_clips = [clip for clip in clip_predictions if clip["prediction"] == 1]

    if positive_clips:
        max_confidence_clip = max(positive_clips, key=lambda x: x["confidence"])
        timestamp = max_confidence_clip["end_time"]
    else:
        timestamp = None

    return {
        "dumping": 1 if timestamp is not None else 0,
        "timestamp": timestamp,
        "clip_predictions": clip_predictions,
    }
