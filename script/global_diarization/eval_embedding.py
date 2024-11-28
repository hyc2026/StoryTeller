import json
import argparse
from get_der import cosine_similarity, calculate_der

def test_prf(total_data, th):
    print('--- similarity prf result ---')
    tot = 0
    pair_data = list()
    for mid in total_data:
        data = total_data[mid]
        for i in range(len(data['start_times'])):
            for j in range(i + 1, len(data['start_times'])):
                pair = dict()
                pair['mid'] = mid
                pair['audio_0'] = i
                pair['audio_1'] = j
                pair['sim_score'] = cosine_similarity(data['embeddings'][i], data['embeddings'][j])
                pair['speaker_0'] = data['gt_speaker'][i]
                pair['speaker_1'] = data['gt_speaker'][j]
                if pair['speaker_0'] == pair['speaker_1']:
                    pair['label'] = 1
                else:
                    pair['label'] = 0
                tot += 1
                pair_data.append(pair)
    print(tot)

    tp = tn = fp = fn = 0
    for data in pair_data:
        if data['sim_score'] >= th and data['label'] == 1:
            tp += 1
        if data['sim_score'] >= th and data['label'] == 0:
            fp += 1
        if data['sim_score'] < th and data['label'] == 1:
            fn += 1
        if data['sim_score'] < th and data['label'] == 0:
            tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("precision=%.2lf" % precision)
    print("recall=%.2lf" % recall)
    print("f1=%.2lf" % (2*precision*recall / (precision+recall)))


def generate_global_id(total_data, global_threshold):
    print('--- global DER result ---')
    
    tot = 0
    tot_der = 0
    
    for mid in total_data:
        data = total_data[mid]
        
        sim_table = list()
        for i in range(len(data['start_times'])):
            sim_table.append(list())
            for j in range(len(data['start_times'])):
                if j < i:
                    sim_table[i].append(sim_table[j][i])
                if j == i:
                    sim_table[i].append(1)
                if j > i:
                    sim_table[i].append(cosine_similarity(data['embeddings'][i], data['embeddings'][j]))

        speaker_global_id = [-1] * len(data['start_times'])
        def find(idx, speaker_id):
            speaker_global_id[idx] = speaker_id
            for i in range(len(data['start_times'])):
                if sim_table[idx][i] >= global_threshold:
                    if speaker_global_id[i] < 0:
                        find(i, speaker_id)
        speaker_cluster_id = 0
        for i in range(len(data['start_times'])):
            if speaker_global_id[i] < 0:
                find(i, speaker_cluster_id)
                speaker_cluster_id += 1

        data['speaker_global_id'] = speaker_global_id

        der = calculate_der(data['start_times'],data['end_times'], data['gt_speaker'], speaker_global_id)
        tot += 1 
        tot_der += der

    print("Total number of mid: %d" % tot)
    print("DER: %.2lf" % (tot_der / tot))

    return total_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--threshold', type=float, default=0.85)
    args = parser.parse_args()
    
    total_data = dict()
    with open(args.input_file) as fin:
        for line in fin:
            data = json.loads(line)
            if "embedding" in data:
                data["embeddings"] = data["embedding"]
            mid = data['mid']
            total_data[mid] = data
            
    test_prf(total_data, args.threshold)
    generate_global_id(total_data, args.threshold)
    
