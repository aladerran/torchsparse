from itertools import repeat
from typing import List, Tuple, Union

import torch
import numpy as np

from functools import wraps
import time

import torchsparse.backend

__all__ = ['make_ntuple', 'timing_decorator', 'TimingManager']


def make_ntuple(x: Union[int, List[int], Tuple[int, ...], torch.Tensor],
                ndim: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())

    assert isinstance(x, tuple) and len(x) == ndim, x
    return x

def make_tensor(x: Tuple[int, ...], dtype: torch.dtype, device) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)

class TimingManager:
    times = {}
    counts = {} 

    @classmethod
    def add_time(cls, name, duration):
        if name not in cls.times:
            cls.times[name] = 0.0
        cls.times[name] += duration

    @classmethod
    def add_count(cls, name): 
        if name not in cls.counts:
            cls.counts[name] = 0
        cls.counts[name] += 1

    @classmethod
    def get_time(cls, name):
        return cls.times.get(name, 0.0)

    @classmethod
    def print_times(cls):
        for name, duration in cls.times.items():
            count = cls.counts.get(name, 0) 
            print(f"{name}: {duration * 1000:.2f}ms, Count: {count}") 

    @classmethod
    def reset_times(cls):
        for name in list(cls.times.keys()): 
            cls.times[name] = 0.0
        for name in list(cls.counts.keys()): 
            cls.counts[name] = 0
    
    def print_backend_profiling_stats():
        torchsparse.backend.print_convolution_forward_time_stats()

def timing_decorator(action_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            TimingManager.add_time(action_name, end_time - start_time)
            TimingManager.add_count(action_name) 
            return result
        return wrapper
    return decorator
