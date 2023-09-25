import numpy as np
import os

from dataset._point_cloud_dataset import get_files

if __name__ == "__main__":
    src_root = "/data/home/aburai/WADS/sequences"
    dst_root = "/data/home/aburai/WADS2/sequences"
    split = [15
    , 18
    , 36
    , 12
    , 17
    , 22
    , 26
    , 28
    , 34
    , 11
    , 16
    , 13
    , 23
    , 14
    , 20
    , 24
    , 30
    , 35
    , 37
    , 76 ]
    im_idx = list()
    for i_folder in split:
        im_idx += get_files('/'.join([src_root, str(i_folder).zfill(2), 'velodyne']), 'bin')
    for im in im_idx:
        raw_data = np.fromfile(im, dtype=np.float32).reshape((-1, 4))
        annotated_data = np.fromfile(im.replace('velodyne', 'labels')[:-3] + 'label',
                                     dtype=np.int32).reshape((-1, 1))
        comb = np.concatenate((raw_data, annotated_data), axis=1)

        unique = np.unique(comb, axis=0)
        u_raw = unique[:,  0:4].reshape(-1).astype(np.float32)
        u_lab = unique[:, 4].reshape(-1).astype(np.int32)
        u_im_file = im.replace("WADS", "WADS2")
        u_lab_file = u_im_file.replace('velodyne', 'labels')[:-3] + 'label'
        os.makedirs(os.path.dirname(u_lab_file), exist_ok=True)
        os.makedirs(os.path.dirname(u_im_file), exist_ok=True)
        u_raw.tofile(u_im_file)
        u_lab.tofile(u_lab_file)
