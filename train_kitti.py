#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix
# ignore weird np warning
import warnings
from deterministic import configure_randomness
from modules import OutDet
from modules.lovasz_losses import lovasz_softmax_flat
from dataset.utils.collate import collate_fn_cp
from dataset import WadsPointCloudDataset

warnings.filterwarnings("ignore")

def main(args):
    data_path = args.data_dir
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    model_save_path = args.model_save_path
    output_model_path = os.path.join(model_save_path, 'KDTree.pt')
    os.makedirs(model_save_path, exist_ok=True)
    device = torch.device(args.device)

    # prepare miou fun
    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)

    class_strings = config["labels"]
    learning_map = config["learning_map"]
    class_inv_remap = config["learning_map_inv"]
    # value for masking labels which are not labeled.
    num_classes = len(class_inv_remap)
    ordered_class_names = [class_strings[class_inv_remap[i]] for i in range(num_classes)]

    epsilon_w = 1e-3
    content = torch.zeros(num_classes, dtype=torch.float, device=device)
    for cl, freq in config["content"].items():
        x_cl = learning_map[cl]  # map actual class to xentropy class
        content[x_cl] += freq
    class_w = content / torch.sum(content)
    loss_w = 1.0 / (class_w + epsilon_w)  # get weights
    # loss_w[0] = 0
    print("Loss weights from content: ", loss_w.data)



    # prepare model
    model = OutDet(num_classes=num_classes, kernel_size=args.K, depth=1)
    # model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # optimizer = QHAdam(model.parameters(), lr=1e-3, nus=(0.7, 1.0), betas=(0.99, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30,40,45], gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss(weight=loss_w)

    mean = [10.88, 0.23, -1.04, 0.21]
    std = [
       11.47,
       6.91,
       0.86,
       0.16]
    # prepare dataset
    train_dataset = WadsPointCloudDataset(device, data_path + '/sequences/', imageset='train',
                                          label_conf=args.label_config, k=args.K*args.K, mean=mean, std=std)
    val_dataset = WadsPointCloudDataset(device, data_path + '/sequences/', imageset='val',
                                        label_conf=args.label_config, k=args.K*args.K, mean=mean, std=std)

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_batch_size,
                                                       shuffle=True,
                                                       num_workers=8,
                                                       collate_fn=collate_fn_cp)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_batch_size,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     collate_fn=collate_fn_cp)

    # training
    epoch = 0
    best_val_miou = 0
    while epoch < args.num_epoch:
        train_loss = train_epoch(epoch, model, train_dataset_loader, criterion, optimizer, num_classes,
                                 ordered_class_names, device)

        val_miou = validate_epoch(epoch, model, val_dataset_loader, criterion, num_classes, ordered_class_names, device)
        scheduler.step()
        # save model if performance is improved
        if best_val_miou <= val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), output_model_path)
        print('Epoch: %d Current val miou is %.3f while the best val miou is %.3f' %
              (epoch, val_miou, best_val_miou))
        epoch += 1

    torch.save(model.state_dict(), os.path.join(model_save_path, 'KDTree_last.pt'))


def train_epoch(epoch, model, train_dataset_loader, criterion, optimizer, n_classes, class_names, device):
    loss_list = []
    pbar = tqdm(total=len(train_dataset_loader), desc=f'Epoch {epoch}')
    # training
    model.train()
    pcm = np.zeros(shape=(n_classes, 2, 2))
    for i_iter, batch in enumerate(train_dataset_loader):
        data = batch['data'][0].to(device)
        ind = batch['ind'][0]
        dist = batch['dist'][0].to(device)
        label = batch['label'][0].long().to(device)
        optimizer.zero_grad()
        logit = model(data, dist, ind)
        loss = criterion(logit, label) + lovasz_softmax_flat(torch.nn.functional.softmax(logit, dim=1),
                                                             label, ignore=None)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        # zero the parameter gradients

        pbar.update(1)
        pbar.set_postfix({'Loss': np.mean(loss_list)})
        predict_labels = torch.argmax(logit, dim=1)
        mcm = multilabel_confusion_matrix(y_true=label.cpu().numpy(), y_pred=predict_labels.cpu().numpy(),
                                          labels=[i for i in range(n_classes)])
        pcm += mcm

    pbar.close()
    IOUs = list()
    for i, cm in enumerate(pcm):
        iou = evaluate_cm(cm, class_names[i])
        IOUs.append(iou)
    class_jaccard = torch.tensor(np.array(IOUs))
    print('Average iou: ', class_jaccard[1:].mean())
    return np.mean(loss_list)


def validate_epoch(epoch, model, val_dataset_loader, criterion, n_classes, class_names, device):
    model.eval()
    val_loss_list = []
    pcm = np.zeros(shape=(n_classes, 2, 2))
    pbar = tqdm(total=len(val_dataset_loader), desc=f'Epoch {epoch}')
    with torch.no_grad():
        for i_iter_val, batch in enumerate(
                val_dataset_loader):
            data = batch['data'][0].to(device)
            ind = batch['ind'][0]
            dist = batch['dist'][0].to(device)
            label = batch['label'][0].long().to(device)
            logit = model(data, dist, ind)
            loss = criterion(logit, label) + lovasz_softmax_flat(torch.nn.functional.softmax(logit, dim=1),
                                                             label, ignore=None)
            predict_labels = torch.argmax(logit, dim=1)
            mcm = multilabel_confusion_matrix(y_true=label.cpu().numpy(), y_pred=predict_labels.cpu().numpy(),
                                              labels=[i for i in range(n_classes)])
            pcm += mcm
            val_loss_list.append(loss.detach().cpu().numpy())
            pbar.update()

            pbar.set_postfix({'Loss': np.mean(val_loss_list)})
    pbar.close()
    IOUs = list()
    for i, cm in enumerate(pcm):
        if i != 0:
            iou = evaluate_cm(cm, class_names[i])
            IOUs.append(iou)
    class_jaccard = torch.tensor(np.array(IOUs))

    val_miou = class_jaccard.sum().item() / class_jaccard.size(0)
    return val_miou


def evaluate_cm(cm, class_name):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    iou1 = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * recall * precision / (precision + recall + 1e-9)
    print(f'Class: {class_name}, Precision:{precision}, Recall: {recall}, IOU: {iou1}, F1: {f1}')
    return iou1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default="/var/local/home/aburai/DATA/SnowyKITTIV2")
    parser.add_argument("--label_config", type=str, default='binary_desnow_kitti.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/var/local/home/aburai/DATA/exp_2023/bin_seg/KDTreeKITTI')
    parser.add_argument('--K', type=int, default=3, help='batch size for training (default: 2)')

    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size for training (default: 2)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation (default: 2)')
    parser.add_argument('--device', type=str, default='cuda:1', help='validation interval (default: 4000)')
    parser.add_argument('--num_epoch', type=int, default=50, help='validation interval (default: 4000)')

    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)
