import numpy as np

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