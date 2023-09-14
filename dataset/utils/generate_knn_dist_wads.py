#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np

import yaml
from tqdm import tqdm
# ignore weird np warning
import warnings
import torch

from dataset.utils.collate import collate_fn_cp
from deterministic import configure_randomness
from dataset import WadsPointCloudDataset


warnings.filterwarnings("ignore")




def main(args):
    data_path = args.data_dir
    train_batch_size = args.train_batch_size
    device = torch.device(args.device)


    # prepare dataset
    tree_k = int(np.round(args.K * args.K))
    train_dataset = WadsPointCloudDataset(device, data_path + '/sequences/', imageset='all',
                                          label_conf=args.label_config, k=tree_k,
                                          shuffle_indices=False, save_ind=True, recalculate=True)

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_batch_size,
                                                       shuffle=False,
                                                       num_workers=0,
                                                       collate_fn=collate_fn_cp)

    pbar = tqdm(total=len(train_dataset))

    for batch in train_dataset_loader:
        pbar.update()
    pbar.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default="/var/local/home/aburai/DATA/WADS2")
    parser.add_argument('--K', type=int, default=3, help='batch size for training (default: 2)')

    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size for training (default: 2)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation (default: 2)')
    parser.add_argument('--device', type=str, default='cuda:0', help='validation interval (default: 4000)')

    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)
