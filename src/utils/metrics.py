def calculate_metrics(clip_outputs):
    TP, FP, FN = 0, 0, 0
    
    for clip_output in clip_outputs:
        preds = clip_output["preds"]
        target = clip_output["targets"]
        video_ids = clip_output["video_ids"]
        start_times = clip_output["start_times"]
        end_times = clip_output["end_times"]
        timestamps = clip_output["timestamps"]
        for i, pred in enumerate(preds):
            if target[i] == pred:
                TP.append(i)
            else:
                if pred == 1 and target[i] == 0:
                    FP += 1
                elif pred == 0 and target[i] == 1:
                    FN += 1
                # this shouldn't happen
                else:
                    FP += 1
                    FN += 1
            
    precision, recall, f1 = 0
    
    precision = TP / TP + FP
    recall = TP / TP + FN    
    f1 = 2 * (precision * recall)/(precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
