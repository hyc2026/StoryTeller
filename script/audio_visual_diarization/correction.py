import json
import os
import re

infer_path = "data/audio_visual_diarization"
correct_threshold = 0.0005
pred_id2name = json.load(open(os.path.join(infer_path, "pred_id2name.json")))
def get_all_jsonl(path):
    result = []
    with open(os.path.join(path, "0.jsonl")) as fp:
        for line in fp.readlines():
            data = json.loads(line)
            result.append(data)
    return result

def product_of_list(lst):
    s = [i[0] for i in lst]
    return sum(s) / len(s)
    # s = 1
    # for i in lst:
    #     s *= i[0]
    # return s

correct_dir = os.path.join(infer_path, "correct")
infer_correct_all = get_all_jsonl(correct_dir)
infer_correct_result = {}

for infer_correct in infer_correct_all:
    data_id = "_".join(infer_correct["id"].split("_")[:-1])
    if data_id not in infer_correct_result:
        infer_correct_result[data_id] = {}
    k, v = infer_correct["text"]["pred_id"][-1]
    if k not in infer_correct_result[data_id]:
        infer_correct_result[data_id][k] = {}
    if v not in infer_correct_result[data_id][k]:
        infer_correct_result[data_id][k][v] = []
    ori_pred_speaker_desc = None
    if v == "Other":
        desc_pattern = k + r": Other #(.*?),"
        matches = re.findall(desc_pattern, infer_correct["text"]["prompt"] + infer_correct["text"]["prediction"])
        if len(matches) == 1:
            ori_pred_speaker_desc = matches[0].strip()
    infer_correct_result[data_id][k][v].append([infer_correct["text"]["score"], infer_correct["id"], ori_pred_speaker_desc])
for k, v in infer_correct_result.items():
    for k1 in v:
        v[k1] = sorted(v[k1].items(), key=lambda x: product_of_list(x[1]), reverse=True)
json.dump(infer_correct_result, open(os.path.join(correct_dir, "infer_correct_result.json"), "w"), indent=4, ensure_ascii=False)

for k, v in infer_correct_result.items():
    for k_id, aligned in v.items():
        aligned_name, aligned_score = aligned[0]
        for score in aligned_score:
            if score[0] > correct_threshold:
                for i in pred_id2name[k][k_id]["gt"][score[1]]:
                    i["aligned_pred_speaker"] = aligned_name
                    i["aligned_pred_speaker_desc"] = score[2]
            else:
                for i in pred_id2name[k][k_id]["gt"][score[1]]:
                    i["aligned_pred_speaker"] = i["ori_pred_speaker"]
                    i["aligned_pred_speaker_desc"] = i["ori_pred_speaker_desc"]
json.dump(pred_id2name, open(os.path.join(correct_dir, "pred_id2name.json"), "w"), indent=4, ensure_ascii=False)

acc, tot, t_acc, t_tot = 0, 0, 0, 0
for _, v in pred_id2name.items():
    for _, i in v.items():
        if len(i["pred"]) > 1:
            for _, j_name in i["gt"].items():
                for j in j_name:
                    if j["speaker"] == j.get("aligned_pred_speaker", j["ori_pred_speaker"]):
                        acc += 1
                    tot += 1
        for _, j_name in i["gt"].items():
            for j in j_name:
                if j["speaker"] == j.get("aligned_pred_speaker", j["ori_pred_speaker"]):
                    t_acc += 1
                t_tot += 1
print("accuracy after align:")
print(t_acc, t_tot, t_acc / t_tot)
print(acc, tot, acc / tot)

test_diarization = {}
for k, v in pred_id2name.items():
    if k not in test_diarization:
        test_diarization[k] = []
    for _, v1 in v.items():
        for _, v3 in v1["gt"].items():
            for v2 in v3:
                if "aligned_pred_speaker" in v2:
                    speaker_gpten_other = v2["aligned_pred_speaker"]
                    speaker_gpten = v2["aligned_pred_speaker_desc"] if v2["aligned_pred_speaker_desc"] is not None else v2["aligned_pred_speaker"]
                else:
                    speaker_gpten_other = v2["ori_pred_speaker"]
                    speaker_gpten = v2["ori_pred_speaker_desc"] if v2["ori_pred_speaker_desc"] is not None else v2["ori_pred_speaker"]
                test_diarization[k].append({
                    "start_time": v2["start_time"],
                    "end_time": v2["end_time"],
                    "speaker_gpten": speaker_gpten,
                    "text_gpten": v2["subtitle"],
                })
for k in test_diarization:
    test_diarization[k] = sorted(test_diarization[k], key=lambda x: x["start_time"])
json.dump(test_diarization, open(os.path.join(correct_dir, "test_diarization.json"), "w"), indent=4, ensure_ascii=False)
