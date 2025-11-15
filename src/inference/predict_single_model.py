import argparse

from src.inference.utils import load_model, load_processor, predict_video


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
