import numpy as np
import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.backbones import SparseResNet21D, SparseResUNet42
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils import TimingManager

import time
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(current_dir, 'output')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)



@torch.no_grad()
def main() -> None:
    # device = 'cuda:0'
    device = 'cpu'
    read_input = False

    backbone = SparseResNet21D
    # backbone = SparseResUNet42

    print(f'{backbone.__name__}:')
    model: nn.Module = backbone(in_channels=4, width_multiplier=1.0)
    model = model.to(device).eval()

    # generate inputs w\ custom batch sizes if not read from file
    if read_input:
        all_coords = torch.from_numpy(np.genfromtxt(f'{output_dir}/coords.csv', delimiter=',')).to(device)
        all_feats = torch.from_numpy(np.genfromtxt(f'{output_dir}/feats.csv', delimiter=',')).to(device)
        input = SparseTensor(coords=all_coords, feats=all_feats).to(device)
    else:
        batch_sizes = 16
        input_size, voxel_size = 500, 0.2
        for batch_size in range(1, batch_sizes+1):
            inputs = np.random.uniform(-100, 100, size=(input_size, 4))
            pcs, feats = inputs[:, :3], inputs
            pcs -= np.min(pcs, axis=0, keepdims=True)
            pcs, indices = sparse_quantize(pcs, voxel_size, return_index=True)
            coords = np.zeros((pcs.shape[0], 4))
            coords[:, :3] = pcs[:, :3]
            coords[:, -1] = batch_size -1
            coords = torch.as_tensor(coords, dtype=torch.int)
            feats = torch.as_tensor(feats[indices], dtype=torch.float)
            if batch_size == 1:
                all_coords = coords
                all_feats = feats
            else:
                all_coords = torch.cat((all_coords, coords), 0)
                all_feats = torch.cat((all_feats, feats), 0)

        # export coords and feats into two csv files
        np.savetxt(f'{output_dir}/coords.csv', all_coords.cpu().numpy(), delimiter=',', fmt='%d')
        np.savetxt(f'{output_dir}/feats.csv', all_feats.cpu().numpy(), delimiter=',', fmt='%f')

    input = SparseTensor(coords=all_coords, feats=all_feats).to(device)

    start_time = time.perf_counter()
    # forward
    outputs = model(input)
    end_time = time.perf_counter()
    print(f"Execution Time: {(end_time - start_time) * 1000} ms")

    # export outputs
    for k, output in enumerate(outputs):
        np.savetxt(f'{output_dir}/out_coords_{backbone.__name__}.csv', output.coords.cpu().numpy(), delimiter=',', fmt='%d')
        np.savetxt(f'{output_dir}/out_feats_{backbone.__name__}.csv', output.feats.cpu().numpy(), delimiter=',', fmt='%f')

    # print feature shapes
    for k, output in enumerate(outputs):
        print(f'output[{k}].F.shape = {output.feats.shape}')

    # print profiling info
    TimingManager.print_times()
    TimingManager.print_backend_profiling_stats()
    TimingManager.reset_times()


if __name__ == '__main__':
    main()
