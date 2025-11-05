import cv2, os, argparse, random
import numpy as np

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your method

################################################

# For all the test videos
for video in os.listdir(args.videos):
    # Process the video
    ret = True
    cap = cv2.VideoCapture(video)
    while ret:
        ret, img = cap.read()
        # Here you should add your code for applying your method

        ########################################################
    cap.release()
    f = open(args.results+video+".txt", "w")
    # Here you should add your code for writing the results
    pos_neg = random.randint(0, 1)
    if pos_neg:
        t = random.randint(0, 10)
        f.write(str(t))
    ########################################################
    f.close()