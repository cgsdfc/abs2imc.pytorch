import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def EPS_like(x: Tensor):
    """
    产生一个EPS数值，放在x相同的设备上。
    """
    return torch.tensor(1e-10, dtype=x.dtype, device=x.device)


def EPS_max(x: Tensor):
    """
    小于EPS的值统一设为EPS，提升数值稳定性。
    """
    return torch.max(x, EPS_like(x))


def convert_tensor(thing, dtype=torch.float, dev="cpu"):
    """
    Convert a np.ndarray or list of them to tensor.
    """
    if isinstance(thing, (list, tuple)):
        return [convert_tensor(x, dtype, dev) for x in thing]
    elif isinstance(thing, np.ndarray):
        return torch.tensor(thing, dtype=dtype, device=dev)
    elif isinstance(thing, torch.Tensor):
        return thing
    elif thing is None:
        return None
    else:
        raise ValueError(f"{type(thing)}")


def convert_numpy(thing):
    """
    Convert a tensor or list of them to numpy.
    """
    if isinstance(thing, (list, tuple)):
        return [convert_numpy(x) for x in thing]
    elif isinstance(thing, torch.Tensor):
        return thing.detach().cpu().numpy()
    elif isinstance(thing, np.ndarray):
        return thing
    elif thing is None:
        return None
    else:
        raise ValueError(f"{type(thing)}")


def default_device():
    """
    返回一个最基本的，大概率可用的设备。
    """
    if torch.cuda.is_available():
        return 0
    return "cpu"
