from itertools import repeat
from typing import List, Tuple, Union

import torch

from functools import wraps
import time

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


class TimingManager:
    times = {}

    @classmethod
    def add_time(cls, name, duration):
        if name not in cls.times:
            cls.times[name] = 0.0
        cls.times[name] += duration

    @classmethod
    def get_time(cls, name):
        return cls.times.get(name, 0.0)

    @classmethod
    def print_times(cls):
        for name, duration in cls.times.items():
            print(f"{name}: {duration * 1000:.2f}ms")

    @classmethod
    def reset_times(cls):
        for name in cls.times.keys():
            cls.times[name] = 0.0
            

def timing_decorator(action_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            TimingManager.add_time(action_name, end_time - start_time)
            return result
        return wrapper
    return decorator