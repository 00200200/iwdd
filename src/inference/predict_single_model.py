import argparse

import torch

from src.inference.utils import extract_clips, load_model, load_processor, process_clip


def predict_video(model, processor, video_path):
    clips = extract_clips(video_path)
    clip_predictions = []
    with torch.no_grad():
        for clip in clips:
            pixel_values = process_clip(clip["frames"], processor)
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

    first_positive_prediction_time = None

    for clip_prediction in clip_predictions:
        if clip_prediction["prediction"] == 1:
            first_positive_prediction_time = clip_prediction["end_time"]
            break

    label = 1 if first_positive_prediction_time is not None else 0

    return {
        "video_label": label,
        "video_timestamp": first_positive_prediction_time,
        "clip_predictions": clip_predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    args = parser.parse_args()

    model = load_model(args.model, args.model_path)
    processor = load_processor(args.model)
    result = predict_video(model, processor, args.video_path)
    print(result)


if __name__ == "__main__":
    main()
