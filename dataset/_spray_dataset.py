
import numpy as np
import os
import torch.utils.data as torch_data
import glob
import torch.utils.data as data
import yaml
import pickle
import torch
import cupy as cp
from cuml import NearestNeighbors


class SemSprayPointCloudDataset(data.Dataset):
    def __init__(self, device, data_path, imageset='train', label_conf='binary_filter_spray.yaml',
                 k=121, leaf_size=100, mean=[0.67974233,  0.4685616,  -0.08652428, 15.3867839],
                 std=[27.67806471, 31.46136095,  0.64550017, 14.13726404],
                 shuffle_indices=False, save_ind=True, recalculate=False):
        self.device = device
        self.recalculate = recalculate
        self.k = k
        self.leaf_size = leaf_size
        self.save_ind = save_ind
        self.mean = np.array(mean)
        self.std = np.array(std)
        with open(label_conf, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.shuffle_indices = shuffle_indices
        self.im_idx = []
        if imageset == 'train':
            meta_file = os.path.join(data_path, "ImageSets", 'train.txt')
            meta = np.loadtxt(meta_file, dtype=str)
        elif imageset == 'val':
            meta_file = os.path.join(data_path, "ImageSets", 'val.txt')
            meta = np.loadtxt(meta_file, dtype=str)
        elif imageset == 'test':
            meta_file = os.path.join(data_path, "ImageSets", 'test.txt')
            meta = np.loadtxt(meta_file, dtype=str)
        elif imageset == 'all':
            meta = list()
            for split in ['train', 'val', 'test']:
                meta_file = os.path.join(data_path, "ImageSets", f'{split}.txt')
                meta += np.loadtxt(meta_file, dtype=str).tolist()
        else:
            raise Exception('Split must be train/val/test')
        for f in meta:
            self.im_idx += list(glob.glob(os.path.join(data_path, f, 'velodyne', '*.bin')))
        self.im_idx.sort()


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 5))[:, :4]

        if self.imageset == 'train':
            if np.random.random() > 0.5:
                rotate_rad = np.deg2rad(np.random.random()*360)
                cos, sine = np.cos(rotate_rad), np.sin(rotate_rad)
                rot_mat = np.matrix([[cos, sine], [-sine, cos]])
                # rotate x and y
                data[:, :2] = np.dot(data[:, :2], rot_mat)
            # if np.random.uniform(high=1.0, low=0.0) >= 0.5:
            #     data[:, 0] *= -1
            # if np.random.uniform(high=1.0, low=0.0) >= 0.5:
            #     data[:, 1] *= -1
            #
            # if np.random.uniform(high=1.0, low=0.0) >= 0.5:
            #     data[:, 2] *= -1
        # data = raw_data


        kd_path = self.im_idx[index].replace('velodyne', 'knn')[:-3] + 'pkl'
        # err = True
        if os.path.exists(kd_path) and not self.recalculate:
            with open(kd_path, 'rb') as f:
                try:
                    ind = pickle.load(f)
                    dist = pickle.load(f)
                    # if ind.shape[1] > self.k:
                    #     ind = ind[:, :self.k]
                    #     dist = dist[:, :self.k]
                    err = False
                except EOFError:
                    err = True
        else:
            err = True
        if err or self.recalculate:
            p1 = torch.from_numpy(data[:, :3]).to(self.device)
            p1 = cp.asarray(p1)
            # metric: string(default='euclidean').
            # Supported
            # distances
            # are['l1, '
            # cityblock
            # ',
            # 'taxicab', 'manhattan', 'euclidean', 'l2', 'braycurtis', 'canberra',
            # 'minkowski', 'chebyshev', 'jensenshannon', 'cosine', 'correlation']
            self.nn = NearestNeighbors()
            while True:

                try:
                    self.nn.fit(p1)
                    break
                except Exception:
                    print("caught it")

            dist, ind = self.nn.kneighbors(p1, self.k)
            # ['euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity']
            # tree = KDTree(data[:, :3], leaf_size=self.leaf_size, metric='cityblock')


            # ind, dist = tree.query_radius(data[:,:3], r=0.5, return_distance=True)
            # process radius and ind for dist query
            # dist = uneven_stack(dist, limit=self.k)
            # ind = uneven_stack(ind, limit=self.k)
            # dist, ind = tree.query(data[:, :3], k=self.k)
            ind = cp.asnumpy(ind)
            dist = cp.asnumpy(dist)
            ind = ind.astype(np.long)
            # dist = dist.reshape(data.shape[0], -1)
            if self.save_ind:
                parent = os.path.dirname(kd_path)
                os.makedirs(parent, exist_ok=True)
                with open(kd_path, 'wb') as f:
                    pickle.dump(ind, f)
                    pickle.dump(dist, f)
        dist = dist + 1.0
        # normalize the distance
        # d_mean = np.mean(dist, axis=1, keepdims=True)
        # d_std = np.std(dist, axis=1, keepdims=True)
        # dist = (dist - d_mean) / d_std
        if self.shuffle_indices:
            s_ind = np.random.rand(*ind.shape).argsort(axis=1)
            ind = np.take_along_axis(ind, s_ind, axis=1)
            dist = np.take_along_axis(dist, s_ind, axis=1)



        annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                      dtype=np.int32).reshape((-1, 1))
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data).reshape(-1)
        data = (data - self.mean) / self.std

        if self.imageset == 'train':
            if np.random.random() > 0.5:
                data[:, :3] += np.random.normal(size=(data.shape[0], 3), loc=0, scale=0.1)
        out_dict = {'data': data.astype(np.float32), 'dist': dist.astype(np.float32), 'ind': ind, 'label': annotated_data.astype(np.uint8)}
        return out_dict
class SemanticSprayDataset(torch_data.Dataset):
    def __init__(self, root_path, split="train"):
        # ----- parse input parameters -------
        self.root_path = root_path
        assert os.path.isdir(self.root_path)
        assert split in ["train", "val", "test", "all"]
        self.training = True if split == "train" else False

        # ------- get data splits -------
        split_dir = os.path.join(self.root_path, "ImageSets", split + ".txt")
        assert os.path.isfile(split_dir)
        bag_list = [x.strip() for x in open(split_dir).readlines()]
        self.lidar_scans = list()
        for bag_path in bag_list:
            bin_files = glob.glob(os.path.join(root_path, bag_path, "velodyne", "*.bin"))
            self.lidar_scans.extend(bin_files)
        self.lidar_scans.sort()
        assert len(self.lidar_scans) > 0


    def load_data(self, scene_path, scan_id):
        assert os.path.isdir(scene_path), print(scene_path)
        data = {}

        # ---------- load top mounted LiDAR point cloud and labels ----------
        # load velodyne:
        velo_path = os.path.join(scene_path, "velodyne", scan_id + ".bin")
        points = np.fromfile(velo_path, np.float32).reshape(-1, 5)  # x,y,z,intensity,ring

        # load labels:
        labels_path = os.path.join(scene_path, "labels", scan_id + ".label")
        labels = np.fromfile(labels_path, np.int32).reshape(-1)  # 0: background, 1: foreground (vehicle), 2: noise

        # ego filter
        box_mask = self.ego_box_filter(points)
        points = points[box_mask]
        labels = labels[box_mask]

        # sanity check
        assert points.shape[0] == labels.shape[0]
        data["points"] = points
        data["labels"] = labels

        # ---------- metadata ----------
        with open(os.path.join(scene_path, "metadata.txt")) as file:
            text_infos = [line.rstrip("\n") for line in file]
        keys = text_infos[0].split(",")
        vals = text_infos[1].split(",")
        assert len(keys) == len(vals)
        metadata = {}
        for k, v in zip(keys, vals):
            metadata[k] = v

        data["infos"] = {"scene_path": scene_path, "scan_id": scan_id, "metadata": metadata}
        return data

    def ego_box_filter(self, points):
        SIZE = 2
        ego_mask = (np.abs(points[:, 0]) < SIZE) & (np.absolute(points[:, 1]) < SIZE)
        return ~ego_mask

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        # --------- get data path ---------
        data_path = os.path.join(self.sample_id_list[index])
        scan_id = data_path.split("/")[-1]
        scene_path = data_path[:-6]

        # --------- load data ---------
        data = self.load_data(scene_path, scan_id)

        return data


if __name__ == "__main__":
    root = '/var/local/home/aburai/DATA/SemantickSpray/SemanticSprayDataset'
    device = torch.device('cuda:0')
    ds = SemSprayPointCloudDataset(device=device, data_path=root, imageset='all', label_conf='../binary_filter_spray.yaml')
    print(len(ds))