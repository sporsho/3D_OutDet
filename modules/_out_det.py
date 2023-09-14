import torch.nn as nn
import yaml
import os
import torch
import numpy as np
import glob

from sklearn.neighbors import KDTree
from torch_cluster import knn_graph




class NHConvBlock(nn.Module):
    def __init__(self, kernel_size=9, in_channels=1, out_channels=8, init=False, dilate=1):
        super().__init__()
        self.ks = kernel_size
        self.dilate = dilate
        self.conv1 = NHConv(kernel_size, in_channels, out_channels, bias=True, init=init, dilate=dilate)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.bn1 = kNNBatchNorm(k=kernel_size, num_feat=out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        # self.pool1 = PoolAvgTree()

        self.conv2 = NHConv(kernel_size, out_channels, out_channels, bias=False, init=False, dilate=dilate)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # self.bn2 = kNNBatchNorm(k=kernel_size, num_feat=out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        # self.pool2 = PoolTree()
        if in_channels != out_channels:
            # self.downsample = ConvTree1D(kernel_size=1, in_channels=in_channels, out_channels=out_channels, bias=True, init=False)
            self.downsample = nn.Conv1d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, bias=False)
            # nn.init.xavier_normal(self.downsample.weight)
            # nn.init.zeros_(self.downsample.bias)
        else:
            self.downsample = None

    def forward(self, data, ind, dist=None):
        inp = data[ind]
        out = self.conv1(inp, dist)
        out = self.bn1(out)
        # out = self.bn1(out, ind)
        out = self.act1(out)
        # out = self.pool1(out, ind)

        out = out[ind]
        out = self.conv2(out, dist)
        out = self.bn2(out)
        # out = self.bn2(out, ind)
        # out = self.pool2(out, ind)

        if self.downsample is not None:
            ds = self.downsample(data.unsqueeze(-1))
        else:
            ds = data
        out = out + ds.squeeze()
        out = self.act2(out)
        return out


class NHConv(nn.Module):
    def __init__(self, kernel_size=9, in_channels=1, out_channels=8, bias=True, init=True, dilate=1):
        super().__init__()
        self.ks = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init = init
        self.bias = bias
        self.dilate = dilate
        if dilate == 2:
            a = [1, 3, 5]
            self.dilate_ind = [8, 10, 12, 22, 24, 26, 36, 38, 40]
        assert dilate <= 2, 'dilation bigger than 2 is not supported right now'
        if init:
            self.in_channels += 1
            if dilate == 2:
                new_size = (np.sqrt(kernel_size) * 2 + 1) ** 2
                self.dw = nn.Parameter(torch.randn(int(new_size)))
            else:
                self.dw = nn.Parameter(torch.randn(kernel_size))
            nn.init.ones_(self.dw)
        self.weight = nn.Parameter(torch.randn(self.in_channels, out_channels, kernel_size))
        nn.init.xavier_normal(self.weight)
        if bias:
            self.b = nn.Parameter(torch.randn(out_channels))
            # nn.init.zeros_(self.b)

    def forward(self, selected_points, dist=None):
        if self.init:
            tmp = self.dw * dist
            data = torch.cat((selected_points, tmp.unsqueeze(-1)), dim=-1)
        else:
            data = selected_points
        if self.in_channels == 1:
            data = data.unsqueeze(-1)
        if self.dilate == 2:
            data = data[:, self.dilate_ind]
        out = torch.einsum("ijk,klj->il", data, self.weight)
        if self.bias:
            out = out + self.b
        return out


class PoolTree(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points, indices):
        selected_points = points[indices]
        out, _ = torch.max(selected_points, dim=1)
        return out


class PoolAvgTree(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points, indices):
        selected_points = points[indices]
        out = torch.mean(selected_points, dim=1)
        return out


class OutDet(nn.Module):

    def __init__(self, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1):
        super().__init__()
        self.pool = pool
        self.num_classes = num_classes
        self.depth = depth
        self.dilate = dilate
        self.tree_kernel = int(np.round(kernel_size * kernel_size))
        self.convs = nn.ModuleList()
        if pool:
            self.pools = nn.ModuleList()
        # self.drops = nn.ModuleList()
        self.drop = nn.Dropout(p=0.5)
        in_channels = 4
        out_channels = 32
        for i in range(depth):
            if i == 0:
                init = True
                suff = 0
            else:
                init = False
                suff = 0
            conv1 = NHConvBlock(kernel_size=self.tree_kernel, in_channels=in_channels + suff,
                                  out_channels=out_channels, init=init, dilate=dilate)
            in_channels = out_channels
            if i < 3:
                out_channels = in_channels * 2
            self.convs.append(conv1)
            if pool and i != depth - 1:
                pool1 = PoolTree()
                self.pools.append(pool1)
            # self.drops.append(nn.Dropout(p=0.5))
        # suff = 3
        self.fc = nn.Linear(in_channels + suff, num_classes, bias=True)
        # nn.init.xavier_normal(self.fc.weight)

    def forward(self, points, dist, indices):
        xyz = points[:, :3].clone()
        out = points
        # out = self.drop(out)
        for i in range(self.depth):
            if i == 0:
                out = self.convs[i](out, indices, dist=dist)
            else:
                out = self.convs[i](out, indices, dist=dist)
            if self.pool and i != self.depth - 1:
                out = self.pools[i](out, indices)
            # out = self.drops[i](out)
            # out = torch.cat((xyz, out), dim=1)
        out = self.drop(out)
        out = self.fc(out)
        return out




if __name__ == '__main__':
    conf_path = '../semantic_wads.yaml'
    with open(conf_path, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    color_map = semkittiyaml['color_map']
    learning_map = semkittiyaml['learning_map']
    labels = semkittiyaml['labels']
    seq_root = '/var/local/home/aburai/DATA/WADS2/sequences'
    seq = '14'
    pc_list = sorted(glob.glob(os.path.join(seq_root, seq, 'velodyne', "*.bin")))
    lab_list = sorted(glob.glob(os.path.join(seq_root, seq, 'labels', "*.label")))
    device = torch.device('cuda:0')
    # model = KDTree().to(device)

    for pc_file, lab_file in zip(pc_list, lab_list):
        labels = np.fromfile(lab_file, dtype=np.int32).reshape(-1)
        pc = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)

        tree = KDTree(pc[:, :3], leaf_size=200)
        dist, ind = tree.query(pc[:, :3], k=9)
        dist = dist.reshape(pc.shape[0], -1)
        dist = dist + 1.0
        labels = labels & 0xFFFF
        labels = np.vectorize(learning_map.__getitem__)(labels)
        label_tensor = torch.from_numpy(labels).long().to(device)
        net = OutDet(kernel_size=3, depth=8).to(device)
        tensor_pt = torch.from_numpy(pc).float().to(device)
        dist = torch.from_numpy(dist).float().to(device)
        ind = torch.from_numpy(ind).to(device)
        logit = net(tensor_pt, dist, ind)
        break
