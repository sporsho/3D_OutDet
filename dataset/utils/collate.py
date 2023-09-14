import numpy as np
import torch


def collate_fn_cp(data):
    data2stack = np.stack([d['data'] for d in data]).astype(np.float32)
    dist2stack = np.stack([d['dist'] for d in data]).astype(np.float32)
    ind2stack = np.stack([d['ind'] for d in data]).astype(np.int64)
    label2stack = np.stack([d['label'] for d in data])

    return {'data': torch.from_numpy(data2stack, ), 'label': torch.from_numpy(label2stack),
            'dist': torch.from_numpy(dist2stack), 'ind': torch.from_numpy(ind2stack)}



def collate_fn_cp_inference(data):
    data2stack = np.stack([d['data'] for d in data]).astype(np.float32)
    dist2stack = np.stack([d['dist'] for d in data]).astype(np.float32)
    ind2stack = np.stack([d['ind'] for d in data]).astype(np.int64)
    # we do not need label during inference, but it does not hurt to provide one
    label2stack = np.stack([d['label'] for d in data])

    return {'data': torch.from_numpy(data2stack, ), 'label': torch.from_numpy(label2stack),
            'dist': torch.as_tensor(dist2stack), 'ind': torch.as_tensor(ind2stack)}




