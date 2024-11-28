import json
import os
import re

infer_path = "data/audio_visual_diarization1"
ori_infer_result = []
with open(os.path.join(infer_path, "0.jsonl")) as f:
    for line in f.readlines():
        ori_infer_result.append(json.loads(line))

pred_id2name = {}
pattern = r"identification\": \"(.*?)\"(, \"recognition|\})"
pattern1 = r"Output order: (.*?)\n"
for ori_infer_clip in ori_infer_result:
    movie_id = "_".join(ori_infer_clip["id"].split("_")[:-1])
    if movie_id not in pred_id2name:
        pred_id2name[movie_id] = {}
    
    matches = re.findall(pattern1, ori_infer_clip["text"]["prompt"])
    if matches[0].strip() == "None":
        pred_order = []
    else:
        pred_order = [j.strip() for j in matches[0].split(",") if j.strip() != ""]
    
    try:
        matches = re.findall(pattern, ori_infer_clip["text"]["prediction"])
        pred_id_name_pairs = matches[0][0].split("#")[:-1]
    except:
        continue
        # print(1)
        # print(json.dumps(ori_infer_clip))
        # break
    if len(pred_order) != len(pred_id_name_pairs):
        # print(2)
        # print(pred_order)
        # print(pred_id_name_pairs)
        print(json.dumps(ori_infer_clip))
        # print("parse error")

    gt_id2name = {}
    for i in ori_infer_clip["text"]["gt"]:

        i["start_time"] = i["start_time"] + ori_infer_clip["video"][0]["start_time"]
        i["end_time"] = i["end_time"] + ori_infer_clip["video"][0]["start_time"]
        if i["speaker_id"] not in gt_id2name:
            gt_id2name[i["speaker_id"]] = []
        gt_id2name[i["speaker_id"]].append(i)

    for pred_id_name_pair in pred_id_name_pairs:
        pred_id, pred_name = pred_id_name_pair.split(":")
        pred_id = pred_id.strip()[-2:]
        assert pred_id in pred_order
        pred_name = pred_name.strip()
        ori_pred_speaker_desc = None
        if pred_name == "Other":
            desc_pattern = pred_id + r": Other # (.*?),"
            matches = re.findall(desc_pattern, ori_infer_clip["text"]["prediction"])
            if len(matches) == 1:
                ori_pred_speaker_desc = matches[0].strip()

        if pred_id not in pred_id2name[movie_id]:
            pred_id2name[movie_id][pred_id] = {"pred": [], "gt": {}}
        if pred_name not in pred_id2name[movie_id][pred_id]["pred"]:
            pred_id2name[movie_id][pred_id]["pred"].append(pred_name)

        pred_id2name[movie_id][pred_id]["gt"][ori_infer_clip["id"]] = []
        for name in gt_id2name[pred_id]:
            pred_id2name[movie_id][pred_id]["gt"][ori_infer_clip["id"]].append({"ori_pred_speaker": pred_name, "ori_pred_speaker_desc": ori_pred_speaker_desc, **name})

acc, tot, t_acc, t_tot = 0, 0, 0, 0
for k in pred_id2name:
    v = sorted(pred_id2name[k].items(), key=lambda x: x[0])
    for i in v:
        if len(i[1]["pred"]) > 1:
            if "Other" not in i[1]["pred"]:
                i[1]["pred"].append("Other")
            for _, j_name in i[1]["gt"].items():
                for j in j_name:
                    if j["speaker"] == j["ori_pred_speaker"]:
                        acc += 1
                    tot += 1
        for _, j_name in i[1]["gt"].items():
            for j in j_name:
                if j["speaker"] == j["ori_pred_speaker"]:
                    t_acc += 1
                t_tot += 1
    pred_id2name[k] = dict(v)
json.dump(pred_id2name, open(os.path.join(infer_path, "pred_id2name.json"), "w"), ensure_ascii=False, indent=4)
print("accuracy before align:")
print(t_acc, t_tot, t_acc / t_tot)
print(acc, tot, acc / tot)

test_diarization = {}
for k, v in pred_id2name.items():
    if k not in test_diarization:
        test_diarization[k] = []
    for _, v1 in v.items():
        for _, v3 in v1["gt"].items():
            for v2 in v3:
                test_diarization[k].append({
                    "start_time": v2["start_time"],
                    "end_time": v2["end_time"],
                    "speaker_gpten": v2["ori_pred_speaker_desc"] if v2["ori_pred_speaker_desc"] is not None else v2["ori_pred_speaker"],
                    "text_gpten": v2["subtitle"],
                })
for k in test_diarization:
    test_diarization[k] = sorted(test_diarization[k], key=lambda x: x["start_time"])
json.dump(test_diarization, open(os.path.join(infer_path, "test_diarization.json"), "w"), indent=4, ensure_ascii=False)

align_data_path = os.path.join(infer_path, "align_data")
os.makedirs(align_data_path, exist_ok=True)   

conflict_id = set()
double_set = set()
pattern = r'([A-E][0-9]):(.*?)#'
with open(os.path.join(align_data_path, "data.jsonl"), "w") as f:
    for ori_infer_clip in ori_infer_result:
        movie_id = "_".join(ori_infer_clip["id"].split("_")[:-1])

        ori_prompt = ori_infer_clip["text"]["prompt"]
        matches = re.findall(pattern1, ori_prompt)
        if matches[0].strip() == "None":
            pred_order = []
        else:
            pred_order = [j.strip() for j in matches[0].split(",") if j.strip() != ""]
        if len(pred_order) == 0:
            continue
        prefix = [" {\"identification\": \""]
        
        for j in pred_order:
            pred_names = pred_id2name[movie_id][j]["pred"]
            if len(pred_names) > 1:
                conflict_id.add(ori_infer_clip["id"])
            prefix_new = []
            for a in prefix:
                for b in pred_names:
                    new_prefix = a + f" {j}: {b} #"
                    if len(pred_names) > 1:
                        if len(prefix) > 1:
                            double_set.add(k)
                        matches = re.findall(pattern, new_prefix)
                        ori_infer_clip["text"]["prompt"] = ori_prompt + new_prefix
                        ori_infer_clip["text"]["pred_id"] = [[x[0], x[1].strip()] for x in matches]
                        f.write(json.dumps(ori_infer_clip, ensure_ascii=False) + '\n')
                    if b != "Other":
                        prefix_new.append(new_prefix)
            prefix = prefix_new
json.dump(list(conflict_id), open(os.path.join(align_data_path, "conflict_id.json"), "w"), ensure_ascii=False, indent=4)