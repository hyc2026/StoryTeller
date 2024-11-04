import json
import os

#original scene split
with open('data/scene_detect/scene_split.json') as fin:
    scene_data = json.loads(fin.read())

#subtitle file
split = json.load(open("data/raw_data/split.json"))
sub_data = {k: v for k, v in json.load(open("data/raw_data/diarization.json")).items() if k.split("_")[1] in split["test_movie_id_list"]}
    
#load original cut
dir = 'data/scene_detect/'
cut_data = dict()
file_names = os.listdir(dir)
for file in file_names:
    if '.csv' not in file:
        continue
    mid = file[:-11]
    cut_data[mid] = list()
    
    with open(dir + file) as fin:
        idx = 0
        for line in fin:
            idx += 1
            if idx <= 2:
                continue
            s = line.split(',')
            cut_data[mid].append(float(s[3]))
    cut_data[mid].append(180.00)

def find_cut(mid, end_time):
    for x in cut_data[mid]:
        if x >= end_time:
            return x
    return end_time

new_scene_data = dict()
for mid in scene_data:
    sub_data[mid].sort(key=lambda x: x['start_time'])
    scene_data[mid].sort(key=lambda x: x['start_time'])
    new_scene_data[mid] = list()
    
    for clip in scene_data[mid]:
        tot_sub = 0
        speakers = set()
        subs = list()
        
        for sub in sub_data[mid]:
            if clip['start_time'] - 0.05 <= sub['start_time'] and sub['end_time'] <= clip['end_time'] + 0.05:
                tot_sub += 1
                speakers.add(sub['speaker'])
                subs.append({'start_time': sub['start_time'], 'end_time': sub['end_time'], 'caption': sub['speaker'] + ': ' + sub['text']})
        
        if len(speakers) == 0 or tot_sub == 1:
            new_scene_data[mid].append(clip)
        else:
            pre_start_time = clip['start_time']
            for idx in range(len(subs) - 1):
                sub = subs[idx]
                if sub['end_time'] - sub['start_time'] <= 0.1:
                    continue
                
                if sub['end_time'] > subs[idx + 1]['start_time']:
                    continue
                    
                end_time = find_cut(mid, sub['end_time'])
                if end_time > subs[idx + 1]['start_time']:
                    end_time = sub['end_time']
                
                new_scene_data[mid].append({'start_time': pre_start_time, 'end_time': end_time})
                pre_start_time = end_time
            
            if clip['end_time'] > pre_start_time:
                new_scene_data[mid].append({'start_time': pre_start_time, 'end_time': clip['end_time']})

no_speaker = 0
one_speaker = 0
more_speaker = 0
for mid in new_scene_data:
    
    for clip in new_scene_data[mid]:
        tot_sub = 0
        speakers = set()
        subs = list()
        
        for sub in sub_data[mid]:
            if clip['start_time'] - 0.05 <= sub['start_time'] and sub['end_time'] <= clip['end_time'] + 0.05:
                tot_sub += 1
                speakers.add(sub['speaker'])
                subs.append({'start_time': sub['start_time'], 'end_time': sub['end_time'], 'caption': sub['speaker'] + ': ' + sub['text']})

json.dump(new_scene_data, open("data/scene_detect/scene_split_new.json", "w"), indent=4, ensure_ascii=False)