import argparse

from src.inference.utils import load_model, load_processor, predict_video


def predict_ensemble(model_results):
    num_clips = len(model_results[0]["clip_predictions"])
    ensemble_clips = []

    for clip_idx in range(num_clips):
        clip_confidence = []
        positive_votes = 0

        for model_result in model_results:
            clip = model_result["clip_predictions"][clip_idx]
            clip_confidence.append(clip["confidence"])
            if clip["prediction"] == 1:
                positive_votes += 1
                confidence += clip["confidence"]

        ensemble_clips.append(
            {
                "prediction": positive_votes,
                "confidence": confidence / positive_votes,
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
            }
        )

    max_positive_votes = max(clip["positive_votes"] for clip in ensemble_clips)
    clips_with_max_votes = [
        clip for clip in ensemble_clips if clip["positive_votes"] == max_positive_votes
    ]

    if max_positive_votes > 0:
        max_confidence_clip = max(clips_with_max_votes, key=lambda x: x["confidence"])

        return {
            "dumping": 1,
            "timestamp": max_confidence_clip["end_time"],
            "clip_predictions": ensemble_clips,
        }
    else:
        return {
            "dumping": 0,
            "timestamp": None,
            "clip_predictions": ensemble_clips,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    args = parser.parse_args()
    models = args.models.split(",")
    checkpoints = args.checkpoints.split(",")
    video_path = args.video_path

    model_results = []
    for model, checkpoint in zip(models, checkpoints):
        model = load_model(model, checkpoint)
        processor = load_processor(model)
        result = predict_video(model, processor, video_path)
        model_results.append(result)

    ensemble_result = predict_ensemble(model_results)
    print(ensemble_result)


if __name__ == "__main__":
    main()
