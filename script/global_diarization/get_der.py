import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from pydub import AudioSegment
import json
import argparse

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

diarizationErrorRate = DiarizationErrorRate()

def calculate_der(start_times, end_times, reference_speaker, predict_speaker):
    reference = Annotation()
    hypothesis = Annotation()
    for i in range(len(start_times)):
        reference[Segment(start_times[i], end_times[i])] = reference_speaker[i]
        hypothesis[Segment(start_times[i], end_times[i])] = predict_speaker[i]
    return diarizationErrorRate(reference, hypothesis)

def get_speaker_num(a):
    return len(set(a))

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # 计算向量的点积
    dot_product = np.dot(vec1, vec2)
    # 计算向量的模长
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    return cosine_sim

def get_one_score(X):
    maximal = 1e6
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            maximal = max(maximal, cosine_similarity(X[i], X[j]))
    return maximal
            

def clustering(X, n_clusters, sample_weight=None):
    kmeans = KMeans(n_clusters)
    if sample_weight is None:
        kmeans.fit(X)
    else:
        kmeans.fit(X, sample_weight=sample_weight)
    y_kmeans = kmeans.predict(X)
    return y_kmeans.tolist()

def clustering_find_k(X, sample_weight=None, one_threshold=0.1, sil_threshold=0.1):
    if len(X) == 1:
        return [0]
                        
    score0 = get_one_score(X)
    if score0 <= one_threshold:
        return [0 for x in X]
    
    if len(X) == 2:
        return [0, 1]
    
    max_score = 0
    rst = None
    for k in range(2, len(X)):
        kmeans = KMeans(k)
        if sample_weight is None:
            kmeans.fit(X)
        else:
            kmeans.fit(X, sample_weight=sample_weight)
        y_kmeans = kmeans.predict(X)
        silhouette_vals = silhouette_samples(X, y_kmeans)
        
        if sample_weight is None:
            score = silhouette_score(X, kmeans.labels_)
        else:
            score = sum([silhouette_vals[idx] * sample_weight[idx] for idx in range(len(sample_weight))]) / sum(sample_weight)
            
        if score > max_score:
            max_score = score
            rst = y_kmeans.tolist()
            
    #print(max_score)
            
    if max_score < sil_threshold:
        return [x for x in range(len(X))]
    
    return rst

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--sample_weight', type=bool, default=False)
    parser.add_argument('--find_k', type=bool, default=False)
    args = parser.parse_args()
    
    tot = 0
    tot_der = 0
    
    
    with open(args.input_file) as fin, open(args.output_file, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            
            if 'embedding' not in data:
                data['embedding'] = data['embeddings']
                
            if args.find_k:
                if args.sample_weight:
                    speaker_rst = clustering_find_k(data['embedding'], sample_weight=data['sample_weight'])
                else:
                    speaker_rst = clustering_find_k(data['embedding'])
            else:
                if args.sample_weight:
                    speaker_rst = clustering(data['embedding'], data['speaker_num'], sample_weight=data['sample_weight'])
                else:
                    speaker_rst = clustering(data['embedding'], data['speaker_num'])
            der = calculate_der(data['start_times'], data['end_times'], data['gt_speaker'], speaker_rst)
            data['speaker_rst'] = speaker_rst
            data['der'] = der
            fout.write(json.dumps(data) + '\n')
            fout.flush()
            tot_der += der
            tot += 1
    print(tot_der / tot)