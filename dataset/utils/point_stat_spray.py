import numpy as np
import glob
import os



def mean_std_spray(spray_root):
    train_meta_file = os.path.join(spray_root, "ImageSets", 'train.txt')
    meta = np.loadtxt(train_meta_file, dtype=str)
    bin_files = list()
    for f in meta:
        bin_files += list(glob.glob(os.path.join(spray_root, f, 'velodyne', '*.bin')))
    # bin_files = glob.glob(os.path.join(stf_root, 'training', subset, '*.bin'))
    point_sum = np.zeros(4)
    n_points = 0
    for bfile in bin_files:
        raw_data = np.fromfile(bfile,
                               dtype=np.float32).reshape((-1, 5))[:, :4]
        point_sum += np.sum(raw_data, axis=0)
        n_points += raw_data.shape[0]

    mu = point_sum / n_points
    squared_diff = np.zeros_like(point_sum)
    for bfile in bin_files:
        raw_data = np.fromfile(bfile,
                               dtype=np.float32).reshape((-1, 5))[:, :4]
        squared_diff += np.sum((raw_data - mu) ** 2, axis=0)
    sigma = np.sqrt(squared_diff / n_points)
    print(mu, sigma)




def count_points_spray(spray_root, num_classes):
    train_meta_file = os.path.join(spray_root, "ImageSets", 'train.txt')
    meta = np.loadtxt(train_meta_file, dtype=str)
    label_files = list()
    for f in meta:
        label_files += list(glob.glob(os.path.join(spray_root, f, 'labels', '*.label')))
    counter = np.zeros(num_classes, dtype=np.int32)
    total = 0
    for lfile in label_files:
        annotated_data = np.fromfile(lfile,
                                     dtype=np.int32).reshape((-1))
        total += annotated_data.shape[0]
        uniques, freqs = np.unique(annotated_data, return_counts=True)
        counter[uniques] += freqs.astype(np.int32)
    for i in range(num_classes):
        print(f'{i}: {counter[i]}')

    assert total == np.sum(counter)


if __name__ == "__main__":
    spray_root = '/var/local/home/aburai/DATA/SemantickSpray/SemanticSprayDataset'

    # count_points_spray(spray_root, 3)
    mean_std_spray(spray_root)

    # spray_mean = [0.67974233,  0.4685616,  -0.08652428, 15.3867839]
    # spray_std = [27.67806471, 31.46136095,  0.64550017, 14.13726404]
