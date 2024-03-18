from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply

from torchsparse.utils import timing_decorator

total_relu_time = 0

__all__ = ['ReLU', 'LeakyReLU']


class ReLU(nn.ReLU):

    @timing_decorator('total_relu_time')
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class LeakyReLU(nn.LeakyReLU):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
