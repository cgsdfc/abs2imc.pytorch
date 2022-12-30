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

import time
from collections import defaultdict
from typing import List

import numpy as np
import sklearn.preprocessing as pp
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
from sklearn.cluster import k_means
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from torch import Tensor

from .idecutils import best_map, cluster_acc
from .idecutils import normalized_mutual_info_score as nmi_score
from .idecutils import purity_score
from .kmeans_pytorch import kmeans as kmeans_torch
from .kmeans_pytorch import pairwise_distance as pairwise_distance_torch
from .torch_utils import convert_numpy

_METRICS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    # Rooted MSE
    "rmse": lambda x, y: np.sqrt(mean_squared_error(x, y)),
}


# 记录当前用到哪些指标，必须和代码实际用到的指标保持同步。
METRICS_LIST = ("ACC", "NMI", "PUR", "F1")


def cluster_f1_score(ytrue, ypred):
    # 因为必须指定 average 的值，参考COMPLETER的做法。

    return f1_score(y_true=ytrue, y_pred=best_map(ytrue, ypred), average="macro")


@torch.no_grad()
def mse_missing_part(X_hat: List[Tensor], X: List[Tensor], M: Tensor):
    loss = 0
    for v in range(len(X_hat)):
        loss += F.mse_loss(X_hat[v][M[:, v]], X[v][M[:, v]])
    loss /= len(X_hat)
    return loss.item()


def imputation_accuracy(Xbar, Xgnd, mask=None, metrics=None, average=False):
    if metrics is None:
        metrics = "mse"
    fn = _METRICS[metrics]
    if isinstance(Xbar, (tuple, list)):
        # Multiview case.
        if mask is None:
            mask = np.ones([Xbar[0].shape[0], len(Xbar)]).astype(bool)
        acc = [fn(xbar[m, :], xgnd[m, :]) for xbar, xgnd, m in zip(Xbar, Xgnd, mask.T)]
        if average:
            acc = sum(acc) / len(acc)
    else:
        assert isinstance(Xbar, np.ndarray)
        # Single view case.
        if mask is None:
            mask = np.ones(Xbar.shape[0]).astype(bool)
        acc = fn(Xbar[mask], Xgnd[mask])
    return acc


def sparseness(x):
    return np.count_nonzero(x) / np.size(x)


class MaxMetrics:
    def __init__(self, **kwds):
        """
        MaxMetrics(acc=True, nmi=True, loss=False) 表示一个指标是越大越好，还是越小越好。
        比如 acc、nmi 是越大越好，loss 是越小越好。
        """

        self._data = defaultdict(lambda: -9999)
        self._greater = defaultdict(lambda: True)
        self._current = None
        self._checkpoint = {}
        # 为了保证顺序，必须对kwds 全部进行设置。
        for key, is_greater in kwds.items():
            self._greater[key] = is_greater
            self._data[key] = -9999 if is_greater else 9999

    def update(self, **kwds) -> dict:
        """
        更新一系列指标的值（取最优），返回那个指标更新了。
        """
        self._current = kwds
        updated = {}
        for key, value in kwds.items():
            gt = self._greater[key]
            if (gt and value > self._data[key]) or (not gt and value < self._data[key]):
                self._data[key] = value
                updated[key] = True
            else:
                updated[key] = False
        return updated

    def report(self, current=True, name=None, compact=False, places=4):
        if current:  # 最近一次 update 的值。
            data = self._current
        elif name is not None:
            data = self._checkpoint[name]
        else:  # 当前的全局最优。
            data = self._data
        data = {k: round(v, places) for k, v in data.items()}
        if compact:
            return list(data.values())
        return data

    def save_current_best(self, name):
        "把当前最优指标保存到checkpoints"
        self._checkpoint[name] = self._data.copy()


def Matrix_Sparseness(x: np.ndarray):
    s1 = np.sum(np.abs(x))  # L1 norm
    s2 = np.sum(np.square(x)) ** 0.5  # L2 norm
    num = np.size(x)
    c = s1 / s2
    sqnum = num**0.5
    a = sqnum - c
    b = sqnum - 1
    return a / b


def compute_inertia(X, centroid):
    dist = pairwise_distance_torch(X, centroid)
    min_dist = torch.min(dist, dim=1)[0]
    inertia = torch.sum(min_dist)
    return inertia


def KMeans_Torch(X: Tensor, *, n_clusters, n_init=20, max_iter=1000, verbose=False):
    """
    Return: centroid, ypred
    """
    inertia_best = None  # 越小越好
    ypred_best = None
    centroid_best = None
    iter_best = None
    for i in range(n_init):
        ypred, centroid = kmeans_torch(
            X,
            num_clusters=n_clusters,
            device=X.device,
            tqdm_flag=False,
            iter_limit=max_iter,
        )
        inertia = compute_inertia(X, centroid)
        if verbose:
            print(f"iter {i:04} inertia {inertia:.4f}")
        if inertia_best is None or inertia < inertia_best:
            iter_best = i
            inertia_best = inertia
            ypred_best = ypred
            centroid_best = centroid

    if verbose:
        print(f"best iter {iter_best:04} inertia {inertia_best:.4f}")
    return centroid_best, ypred_best  # follow sklearn.


def KMeans_Evaluate(
    X,
    data,
    *,
    return_centroid=False,
    n_init=20,
    max_iter=1000,
):
    "这个接口是为了在validation中快速看效果的"
    label = data.Y
    n_clusters = data.clusterNum
    if isinstance(X, np.ndarray):
        centroid, ypred, *_ = k_means(
            X, n_clusters=n_clusters, n_init=n_init, max_iter=max_iter
        )
    else:
        # 就用默认参数就好。n_init, max_iter
        assert isinstance(X, Tensor)
        centroid, ypred = KMeans_Torch(
            X,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
        )
        ypred_tensor = ypred
        ypred = convert_numpy(ypred)

    metrics = dict(
        ACC=cluster_acc(label, ypred),
        NMI=nmi_score(label, ypred),
        PUR=purity_score(label, ypred),
        F1=cluster_f1_score(label, ypred),
    )
    if return_centroid:
        print(metrics)
        return metrics, centroid, ypred_tensor
    return metrics


def get_all_metrics(label, ypred):
    metrics = dict(
        ACC=cluster_acc(label, ypred),
        NMI=nmi_score(label, ypred),
        PUR=purity_score(label, ypred),
        F1=cluster_f1_score(label, ypred),
    )
    return metrics


def SpectralClustering(Z, S, c, type, **kwds):
    """
    fastEIG: Z
    regular: S
    """
    if type == "fastEIG":
        A = Z @ np.diag(1 / np.sqrt(np.sum(Z, 0)))
        # theta 是特征值，B是特征向量。
        Theta, B = eigs(A.T @ A, c, which="LM")
        B = B.real
        Theta = Theta.real
        ff = A @ B @ np.diag(-np.sqrt(Theta))
    elif type == "regular":
        S = 0.5 * (S + S.T)
        S = np.maximum(S, S.T)
        L = np.diag(np.sum(S, 1)) - S
        L = np.maximum(L, L.T)
        Theta, ff = eigs(L, k=c, which="SM")
        ff = ff.real
    elif type == "fastSVD":
        A = Z @ np.diag(1 / np.sqrt(torch.sum(Z, 0)))
        ff, *_ = np.linalg.svd(A)
        if ff.shape[1] > c:
            ff = ff[:, :c]
    else:
        raise NotImplementedError

    ff = pp.normalize(ff)
    # 不能update，否则用户提供的参数就没了。
    ypred = k_means(ff, n_clusters=c, n_init=20, max_iter=1000)
    ypred = ypred[1]  # get the label.

    return ypred, ff


def Safe_Real(x: torch.Tensor):
    if x.is_complex():
        return x.real
    return x


def SpectralClustering_Torch(Z, S, c, type, verbose=False, prune_dim0=False, **kwds):
    """
    fastEIG: Z
    regular: S
    """
    eig_start = time.time()
    if type == "fastEIG":
        A = Z @ torch.diag(1 / torch.sqrt(torch.sum(Z, 0)))
        # theta 是特征值，B是特征向量。
        Theta, B = torch.lobpcg(A.T @ A, k=c, largest=True)
        B = Safe_Real(B)
        Theta = Safe_Real(Theta)
        ff = A @ B @ torch.diag(-torch.sqrt(Theta))
    elif type == "regular":
        S = 0.5 * (S + S.T)
        S = torch.maximum(S, S.T)
        L = torch.diag(torch.sum(S, 1)) - S
        L = torch.maximum(L, L.T)
        Theta, ff = torch.lobpcg(L, k=c, largest=False)
        ff = Safe_Real(ff)
    elif type == "fastSVD":
        A = Z @ torch.diag(1 / torch.sqrt(torch.sum(Z, 0)))
        ff, *_ = torch.svd(A)
        if ff.shape[1] > c:
            if prune_dim0:
                # 谱嵌入的dim0方差很小，鉴别信息较少，去掉他。
                ff = ff[:, 1:c]
            else:
                ff = ff[:, :c]
    else:
        raise NotImplementedError

    ff = F.normalize(ff)
    eig_duration = time.time() - eig_start
    if verbose:
        print(f"eig_duration {eig_duration:.6f}")

    kmeans_start = time.time()
    ypred = KMeans_Torch(ff, n_clusters=c, **kwds)[1]
    kmeans_duration = time.time() - kmeans_start
    if verbose:
        print(f"kmeans_duration {kmeans_duration:.6f}")

    return ypred, ff


def Evaluate_Graph(
    data,
    *,
    Z=None,
    S=None,
    type="fastEIG",
    return_spectral_embedding=False,
    **kwds,
):
    c = data.clusterNum
    label = data.Y
    if isinstance(S, np.ndarray) or isinstance(Z, np.ndarray):
        ypred, ff = SpectralClustering(Z=Z, S=S, c=c, type=type)
    else:
        assert isinstance(S, Tensor) or isinstance(Z, Tensor)
        ypred, ff = SpectralClustering_Torch(Z=Z, S=S, c=c, type=type, **kwds)
        ypred = convert_numpy(ypred)

    metrics = dict(
        ACC=cluster_acc(label, ypred),
        NMI=nmi_score(label, ypred),
        PUR=purity_score(label, ypred),
        F1=cluster_f1_score(label, ypred),
    )
    if return_spectral_embedding:
        ff = convert_numpy(ff)
        return ff
    return metrics


def Seed_All():
    import random

    import numpy as np
    import torch

    seed = 123
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True
