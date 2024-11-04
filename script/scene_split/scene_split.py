import os
import json
from tqdm import tqdm

# check whether t will cut off the segments in t_list
def check_diarization(t, t_list):
    for i in t_list:
        if i["start_time"] <= t < i["end_time"]:
            return False
    return True

# return all the subtitle between the start_time and end_time
def get_audio_clip(start_time, end_time, audio_list):
    res = []
    for audio in audio_list:
        keys = ["start_time", "end_time", "speaker", "text"]
        err = False
        for k in keys:
            if k not in audio:
                err = True
                break
        if err:
            continue
        start = max(audio["start_time"], start_time)
        end = min(audio["end_time"], end_time)
        if start_time <= start < end <= end_time:
            text = audio["text"]
            res.append({
                "start_time": start,
                "end_time": end,
                "actor_name": audio["speaker"],
                "text": text
            })
    return res

split = json.load(open("data/raw_data/split.json"))
data = {k: v for k, v in json.load(open("data/raw_data/diarization.json")).items() if k.split("_")[1] in split["test_movie_id_list"]}
# data = json.load(open("/mnt/bn/lab-agent-hl/heyichen/Audio-Visual/data/movie101/test/audio_diarization/data_all_test.json"))

scene_split = {}
tot, qualified, err = 0, 0, 0

for foldername, subfolders, filenames in os.walk("data/scene_detect"):
    for filename in tqdm(filenames):
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(foldername, filename)
        with open(file_path) as f:
            movie_id = filename.split("-")[0]
            scene_split[movie_id] = []
            accumulate_start_time = 0
            scene_split_end_time = []
            for i, line in enumerate(f.readlines()):
                if i <= 1:
                    continue
                s = line.split(",")
                start_time, end_time, length = float(s[3]), float(s[6]), float(s[9])
                assert -0.01 < end_time - start_time - length < 0.01
                scene_split_end_time.append(end_time)
                if check_diarization(end_time, data[movie_id]):
                    scene_split[movie_id].append([accumulate_start_time, end_time, get_audio_clip(accumulate_start_time, end_time, data[movie_id])])
                    accumulate_start_time = end_time
            if accumulate_start_time != end_time:
                scene_split[movie_id].append([accumulate_start_time, end_time, get_audio_clip(accumulate_start_time, end_time, data[movie_id])])

            while True:
                flag = True
                for i, scene_clip in enumerate(scene_split[movie_id]):
                    if len(scene_split[movie_id]) == 1:
                        break
                    if len(scene_clip[2]) == 0:
                        if i == 0:
                            scene_split[movie_id][i + 1][0] = scene_clip[0]
                            scene_split[movie_id].pop(i)
                        elif i == len(scene_split[movie_id]) - 1:
                            scene_split[movie_id][i - 1][1] = scene_clip[1]
                            scene_split[movie_id].pop(i)
                        else:
                            if (scene_split[movie_id][i + 1][1] - scene_split[movie_id][i + 1][0]) < (scene_split[movie_id][i - 1][1] - scene_split[movie_id][i - 1][0]):
                                scene_split[movie_id][i + 1][0] = scene_clip[0]
                                scene_split[movie_id].pop(i)
                            else:
                                scene_split[movie_id][i - 1][1] = scene_clip[1]
                                scene_split[movie_id].pop(i)
                        flag = False
                        break
                if flag:
                    break
            
            ss = []
            for i, scene_clip in enumerate(scene_split[movie_id]):
                if scene_clip[1] - scene_clip[0] > 15 and len(scene_clip[2]) > 0:
                    slice_num = int((scene_clip[1] - scene_clip[0] + 5) // 10) # 第一个超参数
                    slice_gap = (scene_clip[1] - scene_clip[0]) / slice_num
                    slice_time = {scene_clip[0] + slice_gap * (i + 1): 1000 for i in range(slice_num)}
                    for j in range(len(scene_clip[2])):
                        for k, v in slice_time.items():
                            if abs(scene_clip[2][j]["end_time"] - k) < abs(v - k):
                                slice_time[k] = scene_clip[2][j]["end_time"]
                            if abs(scene_clip[2][j]["start_time"] - k) < abs(v - k):
                                slice_time[k] = scene_clip[2][j]["start_time"]
                    clip_num = set()
                    for k, v in slice_time.items():
                        clip_num.add(v)
                    clip_num = sorted(list(clip_num))
                    accumulate_start_time = scene_clip[0]
                    for j in clip_num:
                        ss.append([accumulate_start_time, j, get_audio_clip(accumulate_start_time, j, data[movie_id])])
                        accumulate_start_time = j
                    if accumulate_start_time != scene_clip[1]:
                        x = get_audio_clip(accumulate_start_time, scene_clip[1], data[movie_id])
                        if len(x) == 0:
                            ss[-1][1] = scene_clip[1]
                        else:        
                            ss.append([accumulate_start_time, scene_clip[1], get_audio_clip(accumulate_start_time, scene_clip[1], data[movie_id])])
                else:
                    ss.append(scene_clip)
            scene_split[movie_id] = ss
            
            ss = []
            for i, scene_clip in enumerate(scene_split[movie_id]):
                if scene_clip[1] - scene_clip[0] > 15 and len(scene_clip[2]) == 1:
                    if scene_clip[2][0]["start_time"] - scene_clip[0] > scene_clip[1] - scene_clip[2][0]["end_time"]:
                        ss.append([scene_clip[0], scene_clip[2][0]["start_time"], get_audio_clip(scene_clip[0], scene_clip[2][0]["start_time"], data[movie_id])])
                        ss.append([scene_clip[2][0]["start_time"], scene_clip[1], get_audio_clip(scene_clip[2][0]["start_time"], scene_clip[1], data[movie_id])])
                    else:
                        ss.append([scene_clip[0], scene_clip[2][0]["end_time"], get_audio_clip(scene_clip[0], scene_clip[2][0]["end_time"], data[movie_id])])
                        ss.append([scene_clip[2][0]["end_time"], scene_clip[1], get_audio_clip(scene_clip[2][0]["end_time"], scene_clip[1], data[movie_id])])
                else:
                    ss.append(scene_clip)
            scene_split[movie_id] = ss

            ss = []
            for i, scene_clip in enumerate(scene_split[movie_id]):
                if scene_clip[1] - scene_clip[0] > 15 and len(scene_clip[2]) == 0:
                    slice_num = int((scene_clip[1] - scene_clip[0] + 5) // 10) # 第一个超参数
                    slice_gap = (scene_clip[1] - scene_clip[0]) / slice_num
                    slice_time = {scene_clip[0] + slice_gap * (i + 1): 0 for i in range(slice_num - 1)}
                    for j in range(len(scene_split_end_time)):
                        for k, v in slice_time.items():
                            if abs(scene_split_end_time[j] - k) < abs(scene_split_end_time[v] - k):
                                slice_time[k] = j
                    # print(slice_time)
                    clip_num = set()
                    for k, v in slice_time.items():
                        clip_num.add(v)
                    clip_num = sorted(list(clip_num))
                    clip_num = [scene_split_end_time[j] for j in clip_num]
                    # print(clip_num)
                    accumulate_start_time = scene_clip[0]
                    for j in clip_num:
                        ss.append([accumulate_start_time, j, []])
                        accumulate_start_time = j
                    if accumulate_start_time != scene_clip[1]:
                        ss.append([accumulate_start_time, scene_clip[1], []])
                else:
                    ss.append(scene_clip)
            scene_split[movie_id] = ss

            while True:
                flag = True
                for i, scene_clip in enumerate(scene_split[movie_id]):
                    if len(scene_split[movie_id]) == 1:
                        break
                    if len(scene_clip[2]) == 0 and scene_clip[1] - scene_clip[0] < 5:
                        if i == 0:
                            scene_split[movie_id][i + 1][0] = scene_clip[0]
                            scene_split[movie_id].pop(i)
                        elif i == len(scene_split[movie_id]) - 1:
                            scene_split[movie_id][i - 1][1] = scene_clip[1]
                            scene_split[movie_id].pop(i)
                        else:
                            if (scene_split[movie_id][i + 1][1] - scene_split[movie_id][i + 1][0]) < (scene_split[movie_id][i - 1][1] - scene_split[movie_id][i - 1][0]):
                                scene_split[movie_id][i + 1][0] = scene_clip[0]
                                scene_split[movie_id].pop(i)
                            else:
                                scene_split[movie_id][i - 1][1] = scene_clip[1]
                                scene_split[movie_id].pop(i)
                        flag = False
                        break
                if flag:
                    break
            
            ss = []
            for i, scene_clip in enumerate(scene_split[movie_id]):
                if i > 0:
                    if scene_clip[1] - ss[-1][0] < 15:
                        ss[-1][1] = scene_clip[1]
                    else:
                        ss.append(scene_clip)
                else:
                    ss.append(scene_clip)
            scene_split[movie_id] = ss

            # statistic
            if scene_split[movie_id][-1][1] != 180:
                scene_split[movie_id][-1][1] = 180
            for i, scene_clip in enumerate(scene_split[movie_id]):
                if i > 0:
                    assert scene_clip[0] - scene_split[movie_id][i - 1][1] < 0.1
                if scene_clip[1] - scene_clip[0] < 20:
                    if len(scene_clip[2]) == 0 and scene_clip[1] - scene_clip[0] < 5:
                        err += 1
                    else:
                        qualified += 1
                tot += 1
            
scene_split_save = {k: [{"start_time": i[0], "end_time": i[1]} for i in v] for k, v in scene_split.items()}
json.dump(scene_split_save, open("data/scene_detect/scene_split.json", "w"), ensure_ascii=False, indent=4)
