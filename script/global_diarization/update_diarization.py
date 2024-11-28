import json
import argparse
from eval_embedding import generate_global_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    
    total_data = dict()
    with open(args.input_file) as fin:
        for line in fin:
            data = json.loads(line)
            if "embedding" in data:
                data["embeddings"] = data["embedding"]
            mid = data['mid']
            total_data[mid] = data
    
    global_data = generate_global_id(total_data, args.threshold)

    split = json.load(open("data/raw_data/split.json"))
    data_all = {k: v for k, v in json.load(open("data/raw_data/diarization.json")).items() if k.split("_")[1] in split["test_movie_id_list"]}
        
    none_num = 0
    tot = 0
    res = {}
    for mid in data_all:
        for x in data_all[mid]:
            global_id = None
            try:
                for i in range(len(global_data[mid]['speaker_global_id'])):
                    if global_data[mid]['start_times'][i] <= x['start_time'] and x['end_time'] <= global_data[mid]['end_times'][i]:
                        global_id = global_data[mid]['speaker_global_id'][i]
                        break
            except:
                # print("no global id", mid)
                pass
            if global_id is None:
                none_num += 1
            x['global_id'] = global_id
        if global_id is not None:
            res[mid] = data_all[mid]

    with open(args.output_file, 'w') as fout:
        fout.write(json.dumps(res, ensure_ascii=False, indent=4))
    