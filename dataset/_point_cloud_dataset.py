import torch
from torch.utils import data
import yaml
import os
import glob
import numpy as np
import pickle
import cupy as cp
from cuml.neighbors import NearestNeighbors
from cuml.common.device_selection import set_global_device_type


set_global_device_type('gpu')


def get_files(folder, ext):
    files = glob.glob(os.path.join(folder, f"*.{ext}"))
    return files


class WadsPointCloudDataset(data.Dataset):
    def __init__(self, device, data_path, imageset='train', label_conf='wads.yaml', k=121, leaf_size=100, mean=[0.3420934,  -0.01516175 ,-0.5889243 ,  9.875928  ], std=[25.845459,  18.93466,    1.5863657, 14.734034 ],
                 shuffle_indices=False, save_ind=True, recalculate=False, desnow_root=None, pred_folder=None,
                                         snow_label=None):
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
        self.desnow_root = desnow_root
        self.pred_folder = pred_folder
        self.snow_label = snow_label
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        elif imageset == 'all':
            split = semkittiyaml['split']['train'] + semkittiyaml['split']['valid'] + semkittiyaml['split']['test']
        elif imageset == 'bug':
            split = ["05"]
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.pred_idx = list()
        for i_folder in split:
            self.im_idx += get_files('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), 'bin')
            if desnow_root is not None:
                assert os.path.exists(desnow_root)
                self.pred_idx += get_files('/'.join([desnow_root, str(i_folder).zfill(2), pred_folder]), 'label')

        self.im_idx.sort()
        if desnow_root is not None:
            self.pred_idx.sort()


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                     dtype=np.int32).reshape((-1, 1))
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
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

        if self.desnow_root is not None:
            if self.pred_idx[index].endswith('.pred'):
                preds = np.fromfile(self.pred_idx[index], dtype=np.int64)
            else:
                preds = np.fromfile(self.pred_idx[index], dtype=np.int32)
            preds = preds.reshape(-1)
            snow_indices = np.where(preds == self.snow_label)[0]

            data = np.delete(data, obj=snow_indices.tolist(), axis=0)
            annotated_data = np.delete(annotated_data, obj=snow_indices.tolist(), axis=0)
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


        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data).reshape(-1)
        data = (data - self.mean) / self.std

        if self.imageset == 'train':
            if np.random.random() > 0.5:
                data[:, :3] += np.random.normal(size=(data.shape[0], 3), loc=0, scale=0.1)
        out_dict = {'data': data.astype(np.float32), 'dist': dist.astype(np.float32), 'ind': ind, 'label': annotated_data.astype(np.uint8)}
        return out_dict


class PointCloudDataset(data.Dataset):
    def __init__(self, data_path, imageset='train', label_conf='wads.yaml'):
        self.mean = np.array([0.43,0.29,-0.67,10.8])
        self.std = np.array([1.17,1.40,0.05,0.97])
        with open(label_conf, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        elif imageset == 'all':
            split = semkittiyaml['split']['train'] + semkittiyaml['split']['valid'] + semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.pred_idx = list()
        for i_folder in split:
            self.im_idx += get_files('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), 'bin')
        self.im_idx.sort()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        data = (raw_data - self.mean) / self.std
        annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                     dtype=np.int32).reshape((-1, 1))
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data).reshape(-1)

        out_dict = {'data': data.astype(np.float32), 'label': annotated_data.astype(np.uint8)}
        return out_dict
