import numpy as np
import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.backbones import SparseResNet21D, SparseResUNet42
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils import TimingManager

import time
import os

def set_affinity(core_id):
    os.sched_setaffinity(0, {core_id})

set_affinity(0) # Set the affinity to core 0

current_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(current_dir, 'output')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)



@torch.no_grad()
def main() -> None:
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    for backbone in [SparseResNet21D, SparseResUNet42]:
        print(f'{backbone.__name__}:')
        model: nn.Module = backbone(in_channels=4, width_multiplier=1.0)
        model = model.to(device).eval()

        # generate data
        input_size, voxel_size = 1000, 0.2
        inputs = np.random.uniform(-100, 100, size=(input_size, 4))
        pcs, feats = inputs[:, :3], inputs
        pcs -= np.min(pcs, axis=0, keepdims=True)
        pcs, indices = sparse_quantize(pcs, voxel_size, return_index=True)
        coords = np.zeros((pcs.shape[0], 4))
        coords[:, :3] = pcs[:, :3]
        coords[:, -1] = 0
        coords = torch.as_tensor(coords, dtype=torch.int)
        feats = torch.as_tensor(feats[indices], dtype=torch.float)
        input = SparseTensor(coords=coords, feats=feats).to(device)

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
        TimingManager.reset_times()


if __name__ == '__main__':
    main()
