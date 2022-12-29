# Ploting related stuffs.

from ..data.dataset import PartialMultiviewDataset
from joblib import Parallel, delayed
from pandas import DataFrame as DF
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Dict

import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_images(X, size: int = 32):
    """从一个特征维度是1024的特征矩阵从提取2维灰度图像"""
    return X.reshape([-1, 32, 32]).transpose([0, 2, 1])


def show_gnd_hat(num: int, x_gnd, x_hat, mask):
    """并排显示真实图像和重建图像，xgnd xhat 是获取的灰度图列表，mask是子集掩码，
    用来划分测试集、训练集等。"""
    row = 2
    col = num
    scale = 2

    image_gnd = get_images(x_gnd)[mask]
    image_hat = get_images(x_hat)[mask]

    sampleNum = len(image_gnd)
    samples = np.arange(sampleNum, dtype=int)
    np.random.shuffle(samples)
    samples = samples[:num]

    plt.figure(figsize=(col * scale, row * scale))
    for i, hat, gnd in zip(range(num), image_hat[samples], image_gnd[samples]):
        plt.subplot(2, num, i + 1)
        plt.imshow(hat)
        plt.title(f"hat {i}")
        plt.subplot(2, num, num + i + 1)
        plt.imshow(gnd)
        plt.title(f"gnd {i}")


def plot_gnd_corrupted_hat(imc_data: PartialMultiviewDataset, X_hat):
    """Display sidebyside x_gnd, x_corrupted and x_hat"""
    rows = imc_data.viewNum
    cols = 3
    scale = 5

    plt.figure(figsize=((cols + 1) * scale, rows * scale))
    for v, (x_gnd, x_corrup, x_hat) in enumerate(
        zip(imc_data.X_gnd, imc_data.X_corrupted, X_hat)
    ):
        k = 3 * v
        sns.heatmap(x_gnd, ax=plt.subplot(rows, cols, k + 1))
        plt.title(f"x_gnd {v}")

        sns.heatmap(x_corrup, ax=plt.subplot(rows, cols, k + 2))
        plt.title(f"x_corrupted {v}")

        sns.heatmap(x_hat, ax=plt.subplot(rows, cols, k + 3))
        plt.title(f"x_hat {v}")


def plot_gnd_hat(X_gnd, X_hat, pdist=False):
    """Display sidebyside x_gnd and x_hat"""
    assert len(X_gnd) == len(X_hat)
    rows = 2
    cols = len(X_gnd)
    scale = 5

    if pdist:  # Plot distance matrix instead.
        dis = Parallel(n_jobs=-1, verbose=1)(
            delayed(pairwise_distances)(x) for x in itertools.chain(X_gnd, X_hat)
        )
        X_gnd, X_hat = dis[:cols], dis[cols:]

    plt.figure(figsize=((cols + 1) * scale, rows * scale))
    for v, (x_gnd, x_hat) in enumerate(zip(X_gnd, X_hat), start=1):
        sns.heatmap(x_gnd, ax=plt.subplot(rows, cols, v))
        plt.title(f"x_gnd {v}")

        sns.heatmap(x_hat, ax=plt.subplot(rows, cols, v + cols))
        plt.title(f"x_hat {v}")


def plot_gnd_corrupted(imc_data: PartialMultiviewDataset):
    """Display sidebyside x_gnd and x_corrupted"""
    rows = 2
    cols = imc_data.viewNum
    scale = 5

    plt.figure(figsize=((cols + 1) * scale, rows * scale))
    for v, (x_gnd, x_corrup) in enumerate(
        zip(imc_data.X_gnd, imc_data.X_corrupted), start=1
    ):
        sns.heatmap(x_gnd, ax=plt.subplot(rows, cols, v))
        plt.title(f"x_gnd {v}")

        sns.heatmap(x_corrup, ax=plt.subplot(rows, cols, v + cols))
        plt.title(f"x_corrupted {v}")


def plot_similarity(imc_data: PartialMultiviewDataset):
    """Display similarity of gnd and corrupted data"""
    rows = 2
    cols = imc_data.viewNum
    scale = 5

    # Compute D for all things.
    D = Parallel(n_jobs=-1)(
        delayed(pairwise_distances)(x, n_jobs=-1)
        for x in itertools.chain(imc_data.X_gnd, imc_data.X_avail)
    )

    plt.figure(figsize=((cols + 1) * scale, rows * scale))
    for v, (x_gnd, x_avail) in enumerate(zip(D[:cols], D[cols:]), start=1):
        sns.heatmap(x_gnd, ax=plt.subplot(rows, cols, v))
        plt.title(f"x_gnd {v}")

        sns.heatmap(x_avail, ax=plt.subplot(rows, cols, v + cols))
        plt.title(f"x_avail {v}")
