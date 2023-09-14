import torch
from ptflops import get_model_complexity_info
from modules import OutDet

def prepare_input(res):
    data = torch.FloatTensor(*res)
    dist = torch.FloatTensor(res[0], 9)
    ind = torch.LongTensor(res[0], 9) % res[0]
    return {"points": data, "dist": dist, "indices": ind}



if __name__ == "__main__":
    device = torch.device('cpu')
    net = OutDet(num_classes=2, depth=1, kernel_size=3).to(device)
    N = 150000
    l1 = N * 4 * 9 * 32
    l2 = N * 32 * 9 * 32
    nh_mac = (l1 + l2) / 1000000
    flops, params = get_model_complexity_info(net, (N, 4), input_constructor=prepare_input, as_strings=True,  print_per_layer_stat=True)
    print(f'Flops: {float(flops[:-4]) + nh_mac} MMac')
    print(f'Params: {params}')
