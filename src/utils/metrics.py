def calculate_metrics(clip_outputs):
    video_data = {}
    delta_t = 3
    t_max = 10
    for clip_output in clip_outputs:
        preds = clip_output["preds"].cpu()
        video_ids = clip_output["video_ids"]
        start_times = clip_output["start_times"]
        end_times = clip_output["end_times"]
        video_labels = clip_output["video_labels"]
        video_timestamps = clip_output["video_timestamps"]

        for pred, video_id, start_time, end_time, video_label, timestamp in zip(
            preds, video_ids, start_times, end_times, video_labels, video_timestamps
        ):
            if video_id not in video_data:
                video_data[video_id] = {
                    "predictions": [],
                    "label": video_label,
                    "timestamp": timestamp,
                }
            video_data[video_id]["predictions"].append(
                (start_time, end_time, pred.item())
            )
    tp, fp, fn = 0, 0, 0

    for video_id, data in video_data.items():
        predictions = sorted(data["predictions"], key=lambda x: x[0])
        label = data["label"]
        timestamp = data["timestamp"]
        p = None

        for start_time, end_time, pred in predictions:
            if pred == 1:
                p = end_time
                break

        if label == 1:
            if p is None:
                fn += 1

            elif (timestamp - delta_t) <= p <= (timestamp + t_max):
                tp += 1

            else:
                fp += 1
        else:
            if p is not None:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {"precision": precision, "recall": recall, "f1": f1}
