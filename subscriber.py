#!/usr/bin/env python
import sklearn.metrics

import rospy
import torch
import numpy as np

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
import time
import cupy as cp
from modules import OutDet
from cuml.neighbors import NearestNeighbors
from cuml.common.device_selection import set_global_device_type
from sklearn.metrics import jaccard_score
set_global_device_type('gpu')


def hello_from_spin_lab(pc_data):
    label_file = '/var/local/home/aburai/DATA/WADS2/sequences/11/labels/039498.label'
    label = np.fromfile(label_file, dtype=np.int32).reshape(-1)
    label[label != 0] = 1
    start = time.time_ns()
    device = torch.device('cuda:0')
    gen = point_cloud2.read_points(pc_data, skip_nans=True)
    data = list(gen)
    data = np.stack(data, axis=0)
    p1 = torch.from_numpy(data[:, :3]).to(device)
    p1 = cp.asarray(p1)
    nn = NearestNeighbors()
    while True:
        try:
            nn.fit(p1)
            break
        except Exception:
            print("caught it")

    dist, ind = nn.kneighbors(p1, 9)
    torch_point_cloud = torch.from_numpy(data).to('cuda:0').float()
    torch_dist = torch.Tensor(dist).to(device)
    torch_ind = torch.Tensor(ind).to(device).long()

    model = OutDet(num_classes=2, kernel_size=3, depth=1, dilate=1).to('cuda:0')

    out = model(torch_point_cloud, torch_dist, torch_ind)
    predict_labels = torch.argmax(out, dim=1)
    score = jaccard_score(y_true=label, y_pred=predict_labels.detach().cpu().numpy())
    elapsed = time.time_ns() - start

    # print(f'header: {pc_data.header}')
    print(f'IOU: {score}, exec_time: {elapsed / 1000000}')


class Subscriber:
    def __init__(self, node_name, subscriber_name):
        self.node = node_name
        self.sub = subscriber_name
        self.nn = NearestNeighbors()
        self.device = torch.device('cuda:0')
        self.model = OutDet(num_classes=2, kernel_size=3, depth=1, dilate=1).to(self.device)
        model_path = '/var/local/home/aburai/3D_OutDet/saved_models/bin_desnow_wads/outdet.pt'
        self.model.load_state_dict(torch.load(model_path))
        label_file = '/var/local/home/aburai/DATA/WADS2/sequences/11/labels/039498.label'
        self.label = np.fromfile(label_file, dtype=np.int32).reshape(-1)
        self.label[self.label != 0] = 1

    def run(self):
        rospy.init_node(self.node, anonymous=True)
        rospy.Subscriber(self.sub, PointCloud2, self.desnow)
        rospy.spin()

    def desnow(self, pc_data):
        start = time.time_ns()
        size = len(pc_data.data) // 16
        xyzi = np.ndarray((size, 4), np.float32, pc_data.data).copy()
        data = torch.from_numpy(xyzi).to(self.device)
        p1 = cp.asarray(data[:, :3])
        try:
            self.nn.fit(p1)
        except Exception:
            print("caught it")
        dist, ind = self.nn.kneighbors(p1, 9)
        # torch_point_cloud = torch.from_numpy(data).to('cuda:0').float()
        torch_dist = torch.Tensor(dist).to(self.device)
        torch_ind = torch.Tensor(ind).to(self.device).long()
        out = self.model(data, torch_dist, torch_ind)
        predict_labels = torch.argmax(out, dim=1)
        score = jaccard_score(y_true=self.label, y_pred=predict_labels.detach().cpu().numpy())
        end = time.time_ns()
        elapsed = end -start
        print(f'IOU: {score}, exec_time: {elapsed / 1000000}')
if __name__ == "__main__":
    # rospy.init_node('sub', anonymous=True)
    # rospy.Subscriber('spin', PointCloud2, hello_from_spin_lab)
    # rospy.spin()
    sub = Subscriber('sub', 'spin')
    sub.run()