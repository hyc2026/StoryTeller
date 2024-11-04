import sys
from script.long_video_description.eval_movie_qa import eval as eval_qa

import os
import argparse

from caption_eval import load_caption_from_dense_caption, load_caption_from_inference_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_caption_path', type=str, required=False, default='')
    parser.add_argument('--out_path', type=str, required=False, default='')
    args = parser.parse_args()
    print(args)

    if os.path.isdir(args.pred_caption_path):
        pred_caption_data = load_caption_from_inference_dir(args.pred_caption_path)
    else:
        pred_caption_data = load_caption_from_dense_caption(args.pred_caption_path)
    print("pred_caption_data cnt:", len(pred_caption_data))

    eval_qa("model", pred_caption_data, args.out_path)
