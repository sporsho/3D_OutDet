#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import numpy as np
import torch
import yaml
# ignore weird np warning
import warnings

from modules import OutDet
from dataset.utils.collate import collate_fn_cp_inference, collate_fn_cp
from dataset import WadsPointCloudDataset

warnings.filterwarnings("ignore")



def main(args):
    data_path = args.data_dir
    train_batch_size = args.train_batch_size
    model_save_path = args.model_save_path
    os.makedirs(model_save_path, exist_ok=True)
    device = torch.device(args.device)

    # prepare miou fun
    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)

    class_inv_remap = config["learning_map_inv"]
    # value for masking labels which are not labeled.
    num_classes = len(class_inv_remap)

    keys = class_inv_remap.keys()
    max_key = max(keys)
    look_up_table = np.zeros((max_key + 1), dtype=np.int32)
    for k, v in class_inv_remap.items():
        look_up_table[k] = v
    # prepare model
    model = OutDet(num_classes=num_classes, kernel_size=args.K, depth=1)
    model = model.to(device)



    start = time.time_ns()
    # prepare dataset
    tree_k = int(np.round(args.K * args.K))
    train_dataset = WadsPointCloudDataset(device, data_path + '/sequences/', imageset='all',
                                          label_conf=args.label_config, k=tree_k, shuffle_indices=False,
                                          save_ind=False, recalculate=True)

    if train_dataset.save_ind:
        collate_fn = collate_fn_cp
    else:
        collate_fn = collate_fn_cp_inference
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       collate_fn=collate_fn)

    for i_iter, batch in enumerate(train_dataset_loader):
        data = batch['data'][0].to(device)
        ind = batch['ind'][0].to(device)
        dist = batch['dist'][0].to(device)
        # label = batch['label'][0].long().to(device)

    end = time.time_ns()
    print(f"data pre-processing time: {(end - start) / (1000000 * len(train_dataset_loader))}")

    train_dataset = WadsPointCloudDataset(device, data_path + '/sequences/', imageset='all',
                                          label_conf=args.label_config, k=tree_k, shuffle_indices=False,
                                          save_ind=False, recalculate=False)

    if train_dataset.save_ind:
        collate_fn = collate_fn_cp
    else:
        collate_fn = collate_fn_cp_inference
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       collate_fn=collate_fn)
    forward_pass_list = list()
    post_process_list = list()
    model.eval()
    with torch.no_grad():
        for i_iter_val, batch in enumerate(
                train_dataset_loader):
            data = batch['data'][0].to(device)
            ind = batch['ind'][0].to(device)
            dist = batch['dist'][0].to(device)
            start_time = time.time_ns()
            logit = model(data, dist, ind)
            forward_pass_list.append(time.time_ns() - start_time)
            start_time = time.time_ns()
            predict_labels = torch.argmax(logit, dim=1)

            pred_np = predict_labels.cpu().numpy().reshape(-1)
            inv_labels = look_up_table[pred_np]
            inv_labels = inv_labels.astype(np.int32)
            post_process_list.append(time.time_ns() - start_time)
    print(f'Avg Forward Pass Time: {np.mean(forward_pass_list) /  1000000} ms ')
    print(f'Avg Post Processing  Time: {np.mean(post_process_list) /  1000000} ms ')



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default="/var/local/home/aburai/DATA/WADS2")
    parser.add_argument("--label_config", type=str, default='binary_desnow_wads.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/var/local/home/aburai/DATA/exp_2023/bin_seg/KDTreeV2')
    parser.add_argument('--K', type=int, default=3, help='batch size for training (default: 2)')

    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size for training (default: 2)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation (default: 2)')
    parser.add_argument('--device', type=str, default='cuda:0', help='validation interval (default: 4000)')

    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    # configure_randomness(12345)
    main(args)

#
# data pre-processing time: 78.1447344059098
# Avg Forward Pass Time: 2.654377582166926 ms
# Avg Post Processing  Time: 0.7023440373250389 ms