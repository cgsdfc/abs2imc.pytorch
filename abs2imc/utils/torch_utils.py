# MIT License

# Copyright (c) 2022 Ao Li, Cong Feng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
