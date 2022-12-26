"""
为matlab代码生成mask文件。免得又写一套代码。
"""
import itertools
import traceback

from dataset import MultiviewDataset, P, np, sio, make_mask
from dataclasses import dataclass
from typing import Set
from joblib import Parallel, delayed, Memory
import json

memory = Memory(location="../Cache")


@dataclass
class DatasetInfo:
    name: str
    sampleNum: int
    viewNum: int


@dataclass
class MaskInfo:
    mask_kind: str
    sampleNum: int
    viewNum: int
    per: int
    repeat: int
    parent: P  # 保存在什么地方。
    data_: np.ndarray = None
    dataname_: str = None

    def basename(self):
        return f"sampleNum={self.sampleNum}-viewNum={self.viewNum}-per={self.per}"

    def __hash__(self):
        # hash 必须考虑所有字段，否则有些不会被生成。
        return hash(self.basename() + self.mask_kind)

    def path(self):
        # datasets/masks/general/sampleNum=100-viewNum=20-per=10.mat
        return self.parent.joinpath(self.basename() + ".mat")

    def save(self):
        sio.savemat(
            file_name=str(self.path()),
            mdict=dict(folds=self.data_),
        )

    def make_mask(self):
        # folds = np.zeros([self.repeat, self.sampleNum, self.viewNum], dtype=bool)
        folds = [None] * self.repeat  # 一个fold是一个元素
        for i in range(self.repeat):
            mask = make_mask(
                paired_rate=self.per / 100,
                sampleNum=self.sampleNum,
                viewNum=self.viewNum,
                kind=self.mask_kind,
            )
            folds[i] = mask.astype(bool)
        self.data_ = np.asarray(folds, dtype=bool)

    def run(self):
        if self.path().exists():
            return
        self.parent.mkdir(exist_ok=True, parents=True)
        print(f"Mask: {self.path()}")
        try:
            self.make_mask()
        except:
            traceback.print_exc()
            print(f"Error in {self}")
        else:
            self.save()


class MaskFile_Generator:
    def __init__(self):
        """
        配置参数写这里。
        """
        self.dataset_root = P("../datasets")
        assert self.dataset_root.is_dir()
        self.mask_kind_list = ["general", "weaker", "partial"]

        # 输出路径
        self.mask_root = self.dataset_root.joinpath(f"masks")
        self.mask_root.mkdir(exist_ok=True, parents=True)

        # 从10到100，间隔10
        self.per_list = list(range(10, 110, 10))
        self.repeat = 10  # 每个Mask有5个fold，相当于5次重复实验。
        self.n_jobs = 4

    def get_available_datasets(self):
        """
        返回有效数据集的主要信息。
        """

        @memory.cache
        def get_data_list(dataset_root: P):
            data_list = []
            for f in dataset_root.glob("*.mat"):
                try:
                    data = MultiviewDataset(datapath=str(f))
                    data_list.append(
                        DatasetInfo(
                            name=data.name,
                            sampleNum=data.sampleNum,
                            viewNum=data.viewNum,
                        )
                    )
                except:
                    print(f"Error loading: {f}")
                    continue
            return data_list

        return get_data_list(self.dataset_root)

    def yield_all_mask_info(self):
        """
        产生所有需要的MaskInfo。datasets下面的所有数据集，每个数据集的所有视角组合（从2视角开始）。
        对齐率PER从10到90，10为间隔。
        """
        for data in self.get_available_datasets():
            for viewNum, per, mask_kind in itertools.product(
                range(2, data.viewNum + 1), self.per_list, self.mask_kind_list
            ):
                mask_info = MaskInfo(
                    sampleNum=data.sampleNum,
                    viewNum=viewNum,
                    per=per,
                    repeat=self.repeat,
                    parent=self.mask_root.joinpath(mask_kind),
                    mask_kind=mask_kind,
                )
                mask_info.dataname_ = data.name
                yield mask_info

    def run(self):
        mask_list = list(self.yield_all_mask_info())
        print(f"MaskNum: {len(mask_list)}")
        Parallel(n_jobs=self.n_jobs, verbose=2)(
            delayed(MaskInfo.run)(mask_info) for mask_info in mask_list
        )


if __name__ == "__main__":
    MaskFile_Generator().run()
