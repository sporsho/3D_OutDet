#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import yaml
import warnings
import glob
warnings.filterwarnings("ignore")



def get_unique_labels(data_dir, split):
    files = glob.glob(os.path.join(data_dir, 'sequences', str(split), 'labels', '*.label'))
    out_dict = dict()
    for f in files:
        data = np.fromfile(f, dtype=np.int32).reshape(-1)
        label, counts = np.unique(data, return_counts=True)
        for l, c in zip(label,counts):
            if l in out_dict.keys():
                out_dict[l] += c
            else:
                out_dict[l] = c

    return out_dict


if __name__ == '__main__':
    # Training settings
    data_dir = '/var/local/home/aburai/DATA/WADS2'
    config_file = 'semantic_wads.yaml'

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    labels_original = config['content'].keys()
    learning_map = config['learning_map']
    inv_learning_map = config['learning_map_inv']
    reduced_map = dict()
    for k, v in learning_map.items():
        reduced_map[k] = inv_learning_map[v]
    train_split = config['split']['train']
    test_split = config['split']['test']
    val_split = config['split']['valid']
    all_split = train_split + test_split + val_split
    string_labels = config['labels']
    contents = np.zeros(shape=(300,))
    label_seq = dict()
    for split in all_split:
        unique_dict = get_unique_labels(data_dir, split)
        for k, v in unique_dict.items():
            contents[k] += v
            if k in label_seq.keys():
                label_seq[k].append(split)
            else:
                label_seq[k] = [split]
        # latex_row = get_latex_row_from_unique_dict(unique_dict, labels_original, reduced_map, string_labels)
    name_row = "\\bfseries{Class Name} & \\bfseries{# Points} & \\bfseries{\%}"
    count_row = ""
    c = 0
    s = np.sum(contents)
    for i in range(300):
        if contents[i] != 0:
            lab = string_labels[i]
            fraction = 100 * (contents[i] / s)
            count_row = count_row+"\n"+ lab +" & " + str(int(contents[i])) + " & " +f'{fraction:0.2f}' + " & " + str(len(label_seq[i])) + '\\\\ \\midrule'
            c +=1
    print(name_row)
    print(count_row)
    print(sum(contents) / 1000000)
    all_split.sort()
    print(all_split)