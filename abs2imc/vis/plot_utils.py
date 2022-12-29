"""
所有关于绘图的实用函数。
"""

import seaborn as sns

# from seaborn import heatmap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from pandas import DataFrame
import sklearn.preprocessing as pp
import pandas as pd

# sns.set_theme()


def heatmap_pairwise_distances(x, metrics=None, title: str = None):
    """
    点对距离热力图，为了观察块对角线。
    """
    if metrics is None:
        metrics = "euclidean"
    sns.heatmap(pairwise_distances(x, metric=metrics))
    if title:
        plt.title(title)


def heatmap(x, title: str = None):
    sns.heatmap(x)
    if title:
        plt.title(title)
