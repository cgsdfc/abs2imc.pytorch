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

from collections import Counter
from pathlib import Path as P
from typing import Any, Dict, List

import h5py
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, normalize, scale

from .get_mask import get_mask


def get_shape(thing):
    """
    递归看一个东西的shape
    """
    if isinstance(thing, (tuple, list)):
        return [x.shape for x in thing]
    return thing.shape


def try_get_item(data: Dict[str, Any], key_list: List[str]):
    for key in key_list:
        if key in data:
            return data[key]
    raise ValueError(f"No such key: {data.keys()}")


NORMALIZE_CLASS = {
    "minmax": pp.MinMaxScaler,
    "center": pp.StandardScaler,
    "l2": pp.Normalizer,
}


def num_per_class(Y) -> List[int]:
    return list(Counter(Y).values())


def filter_dict_value(d, pred):
    return {k: v for k, v in d.items() if pred(v)}


class MultiviewDataset:
    """Basic logic of loading a multiview dataset from .mat file."""

    def __init__(
        self,
        datapath: str,  # full path to mat file.
        normalize: str = None,  # normalization strategy.
        view_ids=None,  # subset of views to use.
    ):
        self.describe_excludes = [
            "X",
            "Y",
            "describe_excludes",
            "cluster_sizes",
            "scalers",
        ]
        datapath = P(datapath)
        # Load raw data from .mat file.
        try:
            data = sio.loadmat(str(datapath), simplify_cells=True)
        except NotImplementedError:
            data = h5py.File(str(datapath))

        # Extract X and Y.
        if "X_names" in data:
            X_names = data["X_names"]
            X = [data[str(name)] for name in X_names]
        else:
            X = list(try_get_item(data, ["X", "data"]))

        if "Y_name" in data:
            Y_name = data["Y_name"]
            Y = data[Y_name]
        else:
            Y = try_get_item(data, ["truth", "Y", "gt", "gnd", "truelabel", "y"])

        # make sure y is zero-based (from matlab often is one-based.)
        if np.min(Y) == 1:
            Y -= 1

        # Type conversion.
        if type(X[0]) != np.ndarray:
            # This may be a sparse matrix.
            X = [x.astype(np.float32).toarray() for x in X]
        else:
            X = [x.astype(np.float32) for x in X]

        # Fix problem of transpose.
        sampleNum = len(Y)
        column_instance = all(x.shape[1] == sampleNum for x in X)
        if column_instance:
            # Make sure data are row-instance.
            X = [x.T for x in X]

        # Perform view selection.
        if view_ids:
            # Select a subset of views.
            X = [X[v] for v in view_ids]

        # Normalize data.
        if normalize is not None:
            scalers = [NORMALIZE_CLASS[normalize]() for _ in X]
            X = [scaler.fit_transform(x) for scaler, x in zip(scalers, X)]
        else:
            scalers = None

        self.X = X
        self.Y = Y
        self.sampleNum = sampleNum
        self.viewNum = len(X)
        self.name = datapath.stem
        self.view_ids = view_ids
        self.view_dims = [x.shape[1] for x in X]
        self.cluster_sizes = num_per_class(Y)
        self.clusterNum = len(np.unique(Y))
        self.normalize = normalize
        self.scalers = scalers
        self.datapath = P(datapath)

    def describe(self):
        return dict(
            filter(
                lambda item: not any(
                    item[0].startswith(s) for s in self.describe_excludes
                ),
                self.__dict__.items(),
            )
        )

    def short_name(self):
        return f"{self.name}-{self.normalize}"


class PartialMultiviewDataset(MultiviewDataset):
    def __init__(
        self,
        datapath: str,
        paired_rate: float,
        normalize: str = None,
        view_ids=None,
        partial_kind=None,  # general, partial (default)
    ):
        super().__init__(datapath=datapath, normalize=None, view_ids=view_ids)
        self.describe_excludes.extend(["mask", "idx"])
        if partial_kind is None:
            partial_kind = "partial"

        mask = make_mask(
            paired_rate=paired_rate,
            sampleNum=self.sampleNum,
            viewNum=self.viewNum,
            kind=partial_kind,
        )

        assert mask.dtype == bool

        mask_sum: np.ndarray = np.sum(mask, 1)
        # 对齐样本
        mask_paired = mask_sum == self.viewNum
        # 存在样本（包括了对齐样本）
        mask_all_list = [msk for msk in mask.T]
        assert all(
            np.count_nonzero(m_all) >= np.count_nonzero(mask_paired)
            for m_all in mask_all_list
        ), "存在样本要多于对齐样本"
        # 非对齐，但存在的样本
        mask_single_list = [
            mask_all_list[v] & np.logical_not(mask_paired) for v in range(self.viewNum)
        ]

        # 对齐样本的下标，按原始顺序排列。
        idx_paired = np.nonzero(mask_paired)[0]
        # 非对齐样本的下标，按原始顺序排列。
        idx_single_list = [np.nonzero(m_single)[0] for m_single in mask_single_list]
        # 所有存在样本的下标，按对齐样本在前，非对齐样本在后的顺序排列。
        idx_all_list = [np.r_[idx_paired, idx] for idx in idx_single_list]

        # 原始完整数据。
        X_gnd = self.X
        # 有缺失的数据。
        X = [
            np.where(np.expand_dims(msk, 1), x, np.zeros_like(x))
            for msk, x in zip(mask.T, X_gnd)
        ]

        if normalize is not None:
            assert self.scalers is None
            self.normalize = normalize
            self.scalers = self.init_scalers()
            for x, idx, scaler in zip(X, idx_all_list, self.scalers):
                x[idx] = scaler.fit_transform(x[idx])
            # 用部分数据估计的scaler来变换完整数据，可能不太好。
            # X_gnd = [scaler.transform(x) for scaler, x in zip(self.scalers, X_gnd)]

        X_all = [x[idx] for idx, x in zip(idx_all_list, X)]
        X_paired = [x[idx_paired] for x in X]  # 注意：idx_paired 不是list。
        # 不成对的样本。
        X_single = [x[idx] for idx, x in zip(idx_single_list, X)]

        self.partial_kind = partial_kind
        self.paired_rate = paired_rate
        self.pairedNum = len(idx_paired)
        self.mask = mask
        self.mask_not = np.logical_not(mask)
        self.mask_sum = mask_sum
        self.X = X
        self.X_list = X  # Alias to X
        self.X_gnd = X_gnd
        self.X_all_list = X_all
        self.X_paired_list = X_paired
        self.X_single_list = X_single
        self.idx_all_list = idx_all_list

    def init_scalers(self):
        return [NORMALIZE_CLASS[self.normalize]() for _ in range(self.viewNum)]

    def inverse_scale(self, X_scaled):
        """
        对数据进行反向scale，即恢复到X_gnd的比例。
        """
        if self.scalers is None:
            print("没有进行normalize，不需要inverse_scale")
            return X_scaled
        return [
            scaler.inverse_transform(x) for scaler, x in zip(self.scalers, X_scaled)
        ]

    def short_name(self):
        return f"{self.name}-{self.paired_rate * 100}-{self.normalize}"


class SingleViewDataset(MultiviewDataset):
    """A single view dataset."""

    def __init__(self, datapath: str, view: int, normalize: str = None) -> None:
        datapath = P(datapath)
        super().__init__(datapath=datapath, normalize=normalize, view_ids=[view])
        self.view = view
        self.view_dim = self.view_dims[0]
        self.X = self.X[0]
        if self.scalers is None:
            self.scaler = None
        else:
            self.scaler = self.scalers[0]
        del self.scalers
        del self.view_dims
        del self.view_ids

    def short_name(self):
        return f"{self.name}-{self.view}-{self.normalize}"


def mask_general_multiview(*, paired_rate: float, sampleNum: int, viewNum: int):
    """
    一般缺失掩码。不会保留成对样本。针对多视角。
    """
    return get_mask(
        view_num=viewNum, data_len=sampleNum, missing_rate=(1 - paired_rate) * viewNum
    ).astype(bool)


def mask_weaker_multiview(*, paired_rate: float, sampleNum: int, viewNum: int):
    """
    较弱缺失掩码。不会保留成对样本。针对多视角。
    """
    return get_mask(
        view_num=viewNum, data_len=sampleNum, missing_rate=(1 - paired_rate)
    ).astype(bool)


def mask_partial_twoviews(*, paired_rate: float, sampleNum: int, viewNum=2, label=None):
    """
    成对缺失掩码，针对两视角。
    """
    samples = np.arange(sampleNum)
    paired, partial = train_test_split(
        samples, train_size=paired_rate, stratify=label, shuffle=True
    )
    p1, p2 = train_test_split(partial, train_size=0.5, shuffle=True)

    def index2mask(idx, total):
        "index 的位置为1，其他为0"
        ans = np.zeros(total, dtype=float)
        ans[idx] = 1.0
        return ans

    paired_mask = index2mask(paired, sampleNum)
    p1_mask = index2mask(p1, sampleNum)
    p2_mask = index2mask(p2, sampleNum)

    paired_mask2v = np.c_[paired_mask, paired_mask]
    partial_mask2v = np.c_[p1_mask, p2_mask]
    mask = paired_mask2v + partial_mask2v

    # Sanity Checks:
    # 1. Paired examples.
    mask_sum = np.sum(mask, axis=1)
    assert np.count_nonzero(mask_sum == 2.0) == len(paired)
    # 2. Partial examples
    assert np.count_nonzero((mask[:, 0] == 1) & (mask[:, 1] == 0)) == len(p1)
    assert np.count_nonzero((mask[:, 0] == 0) & (mask[:, 1] == 1)) == len(p2)

    return mask.astype(bool)


def mask_partial_multiview(*, paired_rate, sampleNum, viewNum):
    """
    按对齐样本率PER 产生多视角Mask
    """
    pairedSmp, partialSmp = train_test_split(
        np.arange(sampleNum),
        train_size=paired_rate,
        shuffle=True,
    )  # 随机切分为两部分：Paired 和 Partial
    # 借用getmask，对partial部分删除0.5的特征，即平均每个样本删除一半的视角。
    # 用getmask的好处是，可以避免产生视角全部缺少的样本。
    # 当PER较大，视角数较多时，这个会报错。
    partial_mask = get_mask(
        view_num=viewNum,
        data_len=len(partialSmp),
        missing_rate=1,
    )
    ind = np.zeros([sampleNum, viewNum])
    ind[pairedSmp, :] = 1
    ind[partialSmp, :] = partial_mask
    return ind.astype(bool)


_MASK_FNS = dict(
    general=mask_general_multiview,
    partial=mask_partial_twoviews,
    weaker=mask_weaker_multiview,
)


def make_mask(*, paired_rate: float, sampleNum: int, viewNum: int, kind: str):
    """
    产生缺失掩码的统一入口函数。kind：
    1. general：每个视角都删除相同比例的特征，只保证每个样本至少有一个视角，不保证对齐性，
    最坏情况下，缺失模式呈现对角线，性能严重下降；
    2. partial：保证一定比例的完整的样本（这些样本有全部视角），因此性能不会受严重影响；
    3. weaker：和general类似，但每个视角删除的样本比例除以了视角个数，因此实际的缺失比例
    比设定的要小，性能不会严重影响。
    """
    if paired_rate == 1.0:
        # 这是完整数据，返回全1即可。
        return np.ones([sampleNum, viewNum], dtype=bool)
    return _MASK_FNS[kind](
        paired_rate=paired_rate, sampleNum=sampleNum, viewNum=viewNum
    )


def index2mask(index, sampleNum) -> np.ndarray:
    mask = np.zeros(sampleNum, dtype=bool)
    mask[index] = True
    return mask


def idx_from_mask(mask: np.ndarray):
    """
    求出每个视角的有效样本的下标。用来将它们的表示放回原来的位置。
    """
    N, V = mask.shape
    index = [np.nonzero(mask[:, v])[0] for v in range(V)]
    return index


def get_avail_data(X_gnd, Y_gnd, index_avail):
    """
    获取有效的X和Y，即观察到的数据、非缺失的数据。
    """
    X_avail = [X[idx, :] for X, idx in zip(X_gnd, index_avail)]
    Y_avail = [Y_gnd[idx] for idx in index_avail]
    return X_avail, Y_avail


def train_test_val_masks(sampleNum: int, train_size: float, val_size: float):
    """
    将一个数据集分为 train，test，val三部分，分别指定每个部分的比例。
    返回各部分的mask和id。
    """
    index_all = np.arange(sampleNum)
    assert 0 <= train_size <= 1
    assert 0 <= val_size <= 1
    test_size = 1 - train_size - val_size
    train_val_split, test_split = train_test_split(
        index_all, train_size=train_size + val_size
    )
    train_size2 = train_size / (1 - test_size)
    train_split, val_split = train_test_split(train_val_split, train_size=train_size2)
    return [
        index2mask(index, sampleNum) for index in (train_split, test_split, val_split)
    ]


make_mask_partial_examples = mask_partial_twoviews

if __name__ == "__main__":
    make_mask(
        paired_rate=0.5,
        kind="general",
        sampleNum=100,
        viewNum=4,
    )
