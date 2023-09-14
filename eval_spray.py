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

from dataset import SemSprayPointCloudDataset
from deterministic import configure_randomness
from modules import OutDet
from dataset.utils.collate import collate_fn_cp, collate_fn_cp_inference
from dataset import WadsPointCloudDataset
import warnings


warnings.filterwarnings("ignore")


def main(args):
    data_path = args.data_dir
    test_batch_size = args.test_batch_size
    model_save_path = args.model_save_path
    device = torch.device('cuda:1')
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
    test_dataset = SemSprayPointCloudDataset(device, data_path, imageset='test',
                                           label_conf=args.label_config, k=tree_k,
                                         )
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
            # frame = test_dataset.data_frame.iloc[i_iter_val]

            logit = model(data, dist, ind)
            # loss = crit(logit, label)

            # label = label.squeeze()
            # logit = logit.squeeze()
            predict_labels = torch.argmax(logit, dim=1)
            pcm = multilabel_confusion_matrix(y_true=label.cpu().numpy(), y_pred=predict_labels.cpu().numpy(),
                                        labels=[i for i in range(num_classes)])

            # evaluate_cm(pcm[1], f'snow: {frame.ID}')
            pcms += pcm
            pred_np = predict_labels.cpu().numpy().reshape(-1)
            inv_labels = look_up_table[pred_np]
            inv_labels = inv_labels.astype(np.int32)
            tmp_fname = test_dataset.im_idx[i_iter_val]
            tmp_fname = tmp_fname.replace("/var/local/home/aburai/DATA/SemantickSpray/SemanticSprayDataset", args.test_output_path)
            tmp_fname = tmp_fname.replace("velodyne", "predictions")
            tmp_fname = tmp_fname[:-3] + 'label'

            # img_file = os.path.join(test_dataset.data_root, frame.IMAGESET, 'velodyne', f'{frame.ID}.bin')
            # path_seq, name = get_seq_name_from_path(img_file)
            # path_name = name + ".label"
            # path = os.path.join(args.test_output_path, "sequences",
            #                     path_seq, "predictions", path_name)
            # os.makedirs(os.path.dirname(tmp_fname), exist_ok=True)
            # inv_labels.tofile(tmp_fname)
            pbar.update(1)
    print('*' * 80)
    print('Evaluation using  multilabel confusion matrix')
    print('*' * 80)
    IOUs = list()
    for i in range(1,num_classes):
        iou = evaluate_cm(pcms[i], ordered_class_names[i])
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
    parser.add_argument('-d', '--data_dir', default='/var/local/home/aburai/DATA/SemantickSpray/SemanticSprayDataset')
    parser.add_argument("-label_config", type=str, default='binary_filter_spray.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/var/local/home/aburai/DATA/3D_OutDet/bin_filter_spray/outdet.pt')
    parser.add_argument('-o', '--test_output_path',
                        default='/var/local/home/aburai/DATA/3D_OutDet/bin_filter_spray/outputs')
    parser.add_argument('-m', '--model', choices=['polar', 'traditional'], default='polar',
                        help='training model: polar or traditional (default: polar)')

    parser.add_argument('--device', type=str, default='cuda:0', help='validation interval (default: 4000)')
    parser.add_argument('--K', type=int, default=3, help='batch size for training (default: 2)')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for training (default: 1)')


    args = parser.parse_args()


    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)
