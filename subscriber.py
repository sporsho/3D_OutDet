#!/usr/bin/env python
import rospy
import torch
import numpy as np
from sensor_msgs.msg import PointCloud2
import time
from modules import OutDet
from sklearn.metrics import jaccard_score

def get_alternative_dist(data, k):
    N = data.shape[0]
    start = np.arange(N)
    end = start + k
    # end[end > data.shape[0] - 1] = data.shape[0] - 1
    ind = np.vstack([np.arange(s, e) for s, e in zip(start, end)])
    ind2 = np.where(ind > N -1)
    ind[ind2] -= k
    indexed_data = data[ind]
    sub_data = indexed_data - np.tile(np.expand_dims(data, -2), (1, k, 1))
    dist = np.sqrt(np.sum(sub_data * sub_data, -1))
    dist = dist + 1.0
    return ind, dist



class Subscriber:
    def __init__(self, node_name, subscriber_name):
        self.node = node_name
        self.sub = subscriber_name
        self.device = torch.device('cuda:0')
        self.model = OutDet(num_classes=2, kernel_size=3, depth=1, dilate=1).to(self.device)
        model_path = '/var/local/home/rais/3D_OutDet/saved_models/bin_desnow_wads/outdet.pt'
        self.model.load_state_dict(torch.load(model_path))
        label_file = '/var/local/home/rais/3D_OutDet/sample_data/sample/sample.label'
        self.label = np.fromfile(label_file, dtype=np.int32).reshape(-1)
        self.label[self.label != 0] = 1
        self.max_points = 250000
        self.ind = self.initialize_index(9)

    def initialize_index(self, k):
        N = self.max_points
        start = torch.arange(N).to(self.device)
        end = start + k
        ind = torch.vstack([torch.arange(s, e) for s, e in zip(start, end)]).to(self.device)
        return ind

    def get_alternative_ind_dist(self, data, k):
        N = data.shape[0]
        ind = self.ind[:N]
        ind2 = torch.where(ind > N - 1)
        ind[ind2] -= k
        indexed_data = data[ind]
        sub_data = indexed_data - torch.tile(torch.unsqueeze(data, -2), (1, k, 1))
        dist = torch.sqrt(torch.sum(sub_data * sub_data, -1))
        dist = dist + 1.0
        return ind, dist


    def run(self):
        rospy.init_node(self.node, anonymous=True)
        rospy.Subscriber(self.sub, PointCloud2, self.desnow)
        rospy.spin()

    def desnow(self, pc_data):
        start = time.time_ns()
        size = len(pc_data.data) // 16
        xyzi = np.ndarray((size, 4), np.float32, pc_data.data).copy()
        data = torch.from_numpy(xyzi).to(self.device)
        # ind, dist = get_alternative_dist(xyzi[:, :3], 9)
        ind, dist = self.get_alternative_ind_dist(data, 9)
        # torch_point_cloud = torch.from_numpy(data).to('cuda:0').float()
        torch_dist = torch.Tensor(dist).to(self.device)
        torch_ind = torch.Tensor(ind).to(self.device).long()
        out = self.model(data, torch_dist, torch_ind)
        predict_labels = torch.argmax(out, dim=1)
        score = jaccard_score(y_true=self.label, y_pred=predict_labels.detach().cpu().numpy())
        end = time.time_ns()
        elapsed = end - start
        print(f'IOU: {score}, exec_time: {elapsed / 1000000} millisecond')
if __name__ == "__main__":
    sub = Subscriber('sub', 'OutDet')
    sub.run()