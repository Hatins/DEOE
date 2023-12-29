from pathlib import Path
import argparse
import numpy as np
import ipdb
import heapq

def compute_score(label_file):
    labels_information = np.load(label_file)
    confidence_list = np.array([item[6] for item in labels_information])
    class_list = np.array([item[5] for item in labels_information])
    class_counts = np.bincount(class_list)
    return confidence_list.mean(), class_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--type', default='event',choices=['event', 'frame'], type=str)
    parser.add_argument('--path', default='/data1/gen4_h5_event', type=str)
    parser.add_argument('--topk', default=100, type=int)

    args = parser.parse_args()
    if args.type == 'event':
        Raw_Event_H5_Path = Path(args.path)

    contents = Raw_Event_H5_Path.iterdir()
    
    for dir in contents:
        assert dir.is_dir()
        files = [item for item in dir.iterdir() if item.is_file() and item.suffix == ".npy"]
        scores = dict()
        for label_file in files:
            score, count = compute_score(label_file)
            scores[str(label_file)] = [score,count]
            
        top_scores = heapq.nlargest(args.topk, scores.items(), key=lambda item: item[1][0])  
        ipdb.set_trace()
        total_counts = [sum(item[1][1][0] for item in top_scores)]    
        total_counts = [sum(item[1][1][i] for item in top_scores) for i in range(7)]         


