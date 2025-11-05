import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()

    videos = args.videos
    results = args.results

    pass


if __name__ == "__main__":
    main()
