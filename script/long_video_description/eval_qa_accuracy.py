import json
from script.long_video_description.eval_movie_qa import eval as eval_qa
import os
import argparse

def convert_seconds_to_MMSS(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02}:{int(seconds):02}"

def dense_caption_to_caption(dense_caption):
    result = {}
    for video_id in dense_caption:
        if isinstance(dense_caption[video_id], str):
            result[video_id] = dense_caption[video_id]
            continue
        caption = []
        for event in sorted(dense_caption[video_id], key=lambda x: x['start_time']):
            if 'end_time' not in event or 'start_time' not in event or 'text' not in event or not isinstance(event['text'], str):
                continue
            caption.append(f"{convert_seconds_to_MMSS(event['start_time'])}~{convert_seconds_to_MMSS(event['end_time'])} {event['text']}")
        result[video_id] = "\n".join(caption)
    return result


def load_caption_from_dense_caption(path):
    dense_caption = json.load(open(path, "r"))
    return dense_caption_to_caption(dense_caption)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_caption_path', type=str, required=False, default='')
    parser.add_argument('--out_path', type=str, required=False, default='')
    args = parser.parse_args()
    pred_caption_data = load_caption_from_dense_caption(args.pred_caption_path)
    print("pred_caption_data cnt:", len(pred_caption_data))
    eval_qa("model", pred_caption_data, args.out_path)
