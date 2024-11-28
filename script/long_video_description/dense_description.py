import json

dense_caption = {}
data_err = 0

with open("data/long_video_description/0.jsonl") as f:
    for line in f.readlines():
        data = json.loads(line)


        data_id = "_".join(data["id"].split("_")[:-1])
        if data_id not in dense_caption:
            dense_caption[data_id] = []
        
        start_time = data["video"][0]["start_time"]
        end_time = data["video"][0]["end_time"]

        try:
            prediction = json.loads(data["text"]["prediction"])
            clip_pred_caption = {
                "start_time": start_time,
                "end_time": end_time,
                "text": prediction["caption"]
            }
        except:
            clip_pred_caption = {
                "start_time": start_time,
                "end_time": end_time,
                "text": data["text"]["prediction"]
            }
            data_err += 1
        dense_caption[data_id].append(clip_pred_caption)
print("err count:", data_err)

json.dump(dense_caption, open("result/tarsier/dense_caption_name.json", "w"), ensure_ascii=False, indent=4)

