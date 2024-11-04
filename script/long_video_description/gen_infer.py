import json
import random
import os
from tqdm import tqdm

prompt = """Given a video and supplementary information, please finish the following tasks.
Video: <video>
Known characters: <video> {character}.
Subtitles of the input video with time stamps, and each speaker is identified by a unique ID:
{subtitle}
Tasks:
1. Describe the events that occur within this video."""

audio_diarization = json.load(open("data/audio_visual_diarization/correct/test_diarization.json"))
frame_path = "data/frame"
scene_split = json.load(open("data/scene_detect/scene_split_new.json"))
audio_path = "data/audio"
actor_path = "data/raw_data/ref_actor"

# IDs that may be used
all_ids = [f"{i}{j}" for i in ["A", "B", "C", "D", "E"] for j in range(10)]
def compare_id(item):
    return item[0], item[1]

# Return all diarization included in the event from the 3-minute clip
def get_audio_clip(start_time, end_time, audio_list):
    res = []
    for audio in audio_list:
        start = max(audio["start_time"], start_time) - start_time
        end = min(audio["end_time"], end_time) - start_time
        if 0 <= start < end <= end_time - start_time:
            if end - start < 1 and (start == 0 and audio["start_time"] < start_time or end_time < audio["end_time"] and end == end_time - start_time):
                continue
            res.append({
                "start_time": start,
                "end_time": end,
                "speaker": audio["speaker"],
                "text": audio["text"],
            })
    return res

def gen_data_item():
    f = open("data/long_video_description/data.jsonl", "w")
    for k, v in tqdm(audio_diarization.items()):

        # character table
        ref_actor_box, ref_actor_info, idx, have_actor = {}, {"image_file": {"hl": []}}, 0, False
        while True:
            if os.path.exists(os.path.join(actor_path, f"{k}_ref{idx}.jpg")):
                have_actor = True
                ref_actor_info["image_file"]["hl"].append(os.path.join(actor_path, f"{k}_ref{idx}.jpg"))
                boxes = json.load(open(os.path.join(actor_path, f"{k}_ref{idx}.json")))
                for actor_name_k, bounding_box_v in boxes.items():
                    ref_actor_box[actor_name_k] = bounding_box_v
                idx += 1
            else:
                break
        ref_actor_info["n_frames"] = idx
        
        # subtitle
        for idx, event in enumerate(scene_split[k]):
            audio_clips = get_audio_clip(event["start_time"], event["end_time"], v)

            # 生成输入subtitle
            clip_sub_titles = []
            for audio_clip in audio_clips:
                clip_sub_titles.append("{} - {} ".format(round(audio_clip["start_time"], 2), round(audio_clip["end_time"], 2)) + audio_clip["speaker"] + ": " + audio_clip["text"])

            video_file = {
                "image_file": {
                    "hl": [os.path.join(frame_path, f"{k}_{idx}_{num}.jpg") for num in range(8)]
                },
                "start_time": max(event["start_time"], 0),
                "end_time": min(event["end_time"], 180),
                "n_frames": 8
            }
            
            res = {
                "id": k,
                "text": {
                    "prompt": prompt,
                },
                "video": [
                    video_file
                ]
            }
            if have_actor:
                res["video"].append(ref_actor_info)
            
            res["audio"] = [
                {
                    "audio_file": {
                        "hl": [
                            os.path.join(audio_path, k + ".wav")
                        ]
                    },
                    "start_time": [
                        max(event["start_time"], 0)
                    ],
                    "end_time": [
                        min(event["end_time"], 180)
                    ]
                }
            ]

            res["text"]["prompt"] = res["text"]["prompt"].format(character=json.dumps(ref_actor_box, ensure_ascii=False), subtitle="\n".join(clip_sub_titles))
            
            if not have_actor:
                res["text"]["prompt"] = res["text"]["prompt"].replace("Known characters: <video>", "Known characters:")
            res["text"]["prompt"] = "USER: " + res["text"]["prompt"].strip() + '\nASSISTANT: '

            res["text"]["prompt"] = res["text"]["prompt"].strip()
            res["id"] += "_" + str(idx)
            
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    gen_data_item()
