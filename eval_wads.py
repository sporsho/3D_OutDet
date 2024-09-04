#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import sys
import numpy as np
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import multilabel_confusion_matrix

from deterministic import configure_randomness
from modules import OutDet
from dataset.utils.collate import collate_fn_cp, collate_fn_cp_inference
from dataset import WadsPointCloudDataset
import warnings


warnings.filterwarnings("ignore")





def get_seq_name_from_path(path):
    tmps = path.split(os.path.sep)
    seq = tmps[-3]
    name = tmps[-1]
    tmps2 = name.split(".")
    name = tmps2[0]
    return seq, name


def main(args):
    data_path = args.data_dir
    test_batch_size = args.test_batch_size
    model_save_path = args.model_save_path
    device = torch.device(args.device)
    dilate = 1

    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)

    class_strings = config["labels"]
    class_inv_remap = config["learning_map_inv"]
    num_classes = len(class_inv_remap)

    keys = class_inv_remap.keys()
    max_key = max(keys)
    look_up_table = np.zeros((max_key + 1), dtype=np.int32)
    for k, v in class_inv_remap.items():
        look_up_table[k] = v

    ordered_class_names = [class_strings[class_inv_remap[i]] for i in range(num_classes)]
    # prepare model
    model = OutDet(num_classes=num_classes, kernel_size=args.K, depth=1, dilate=dilate)
    model = model.to(device)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    else:
        raise ValueError()

    # prepare dataset
    tree_k = int(np.round(args.K * args.K))
    test_dataset = WadsPointCloudDataset(device, data_path + '/sequences/', imageset='test',
                                           label_conf=args.label_config, k=tree_k,
                                         desnow_root=args.desnow_root, pred_folder=args.pred_folder,
                                         snow_label=args.snow_label, recalculate=True, save_ind=False)
    if test_dataset.save_ind:
        collate_fn = collate_fn_cp
    else:
        collate_fn = collate_fn_cp_inference
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      collate_fn=collate_fn)

    # validation
    print('*' * 80)
    print('Test network performance on validation split')
    print('*' * 80)
    pbar = tqdm(total=len(test_dataset_loader))
    model.eval()

    pcms = np.zeros(shape=(num_classes, 2, 2), dtype=np.int64)
    with torch.no_grad():
        for i_iter_val, batch in enumerate(
                test_dataset_loader):
            data = batch['data'][0].to(device)
            ind = batch['ind'][0]
            dist = batch['dist'][0].to(device)
            label = batch['label'][0].long().to(device)
            logit = model(data, dist, ind)
            # loss = crit(logit, label)

            # label = label.squeeze()
            # logit = logit.squeeze()
            predict_labels = torch.argmax(logit, dim=1)
            pcm = multilabel_confusion_matrix(y_true=label.cpu().numpy(), y_pred=predict_labels.cpu().numpy(),
                                        labels=[i for i in range(num_classes)])

            # evaluate_cm(pcm[1], f'snow: {i_iter_val}')
            pcms += pcm
            pred_np = predict_labels.cpu().numpy().reshape(-1)
            inv_labels = look_up_table[pred_np]
            inv_labels = inv_labels.astype(np.int32)
            if args.save_pred:
                path_seq, name = get_seq_name_from_path(test_dataset.im_idx[i_iter_val])
                path_name = name + ".label"
                path = os.path.join(args.test_output_path, "sequences",
                                    path_seq, "predictions", path_name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                inv_labels.tofile(path)
            pbar.update(1)
    print('*' * 80)
    print('Evaluation using  multilabel confusion matrix')
    print('*' * 80)
    IOUs = list()

    for i in range(0,num_classes):
        iou = evaluate_cm(pcms[i], ordered_class_names[i])
        print(pcms[i])
        IOUs.append(iou)
    class_jaccard = torch.tensor(np.array(IOUs))
    m_jaccard = class_jaccard.mean().item()

    for i, jacc in enumerate(class_jaccard):
        sys.stdout.write('{jacc:.2f} &'.format(jacc=jacc.item() * 100))
        # sys.stdout.write('{jacc:.2f}\\% &'.format(jacc=jacc.item() * 100))
        # sys.stdout.write(",")
    sys.stdout.write('{jacc:.2f}'.format(jacc=m_jaccard * 100))
    # sys.stdout.write(",")
    # sys.stdout.write('{acc:.2f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    for i in range(1,len(class_jaccard)):
        sys.stdout.write('\\bfseries{{ {name} }} &'.format(name=ordered_class_names[i]))
    sys.stdout.write('\n')
    sys.stdout.flush()


def evaluate_cm(cm, class_name):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    iou1 = tp / (tp + fp + fn)
    f1 = 2 * recall * precision / (precision + recall)
    print(f'Class: {class_name}, Precision:{precision}, Recall: {recall}, IOU: {iou1}, F1: {f1}')
    return iou1


if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='/var/local/home/aburai/DATA/WADS2')
    parser.add_argument("-label_config", type=str, default='binary_desnow_wads.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/var/local/home/aburai/DATA/3D_OutDet/bin_desnow_wads/outdet.pt')
    parser.add_argument('-o', '--test_output_path',
                        default='/var/local/home/aburai/DATA/3D_OutDet/bin_desnow_wads/outputs')

    parser.add_argument('--device', type=str, default='cuda:0', help='validation interval (default: 4000)')
    parser.add_argument('--K', type=int, default=3, help='batch size for training (default: 2)')

    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for training (default: 1)')

    parser.add_argument(
        '--desnow_root', '-dr',
        type=str,
        default=None,
        help='Set this if you want to use the Uncertainty Version'
    )
    parser.add_argument("--pred_folder",
                        type=str,
                        default=None)
    parser.add_argument('--snow_label',
                        type=int,
                        default=None)

    parser.add_argument('--save_pred', type=bool, default=False)
    args = parser.parse_args()


    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)
