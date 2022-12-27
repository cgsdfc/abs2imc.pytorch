import json
import os
from turtle import forward
from typing import List
from pathlib import Path as P

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import argparse
import logging
import math
import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# TODO: don't import *
from abss_imc.data import PartialMultiviewDataset
from abss_imc.model import *
from abss_imc.utils import *
from abss_imc.vis import *

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--per", type=float, default=0.5, help="paired example rate")
parser.add_argument("--ppl", type=int, default=12, help="perplexity")
parser.add_argument("--lamda", type=float, default=0.1, help="lambda")
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--datapath", type=str, default="./datasets/coil20_3view_1024.mat")
parser.add_argument("--views", type=str, default="0,1", help="view ids")
parser.add_argument("--savedir", type=str, default="./output/debug")
parser.add_argument("--hidden_dims", type=int, default=128, help="dims of hidden rep")
parser.add_argument("--logfile", type=str, default="./train.log")
parser.add_argument("--eval_epochs", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1, help="completer train epochs")
parser.add_argument("--cluster_update_epoch", type=int, default=10)
parser.add_argument("--use_mlp", type=bool, default=True, help="use mlp encoder")
parser.add_argument(
    "--stop_thresh",
    type=float,
    default=1,
    help="stop when label change diff is lower than this value",
)

args = parser.parse_args()
args.views = [int(x) for x in args.views.split(",")]
args.savedir = P(args.savedir)
args.savedir.mkdir(parents=1, exist_ok=1)
args.datapath = P(args.datapath)
assert args.datapath.exists()
args.logfile = args.savedir.joinpath("train.log")

seed = args.seed
np.random.seed(seed)
random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.cuda.manual_seed(seed + 3)
torch.backends.cudnn.deterministic = True

args.device = "cpu" if not torch.cuda.is_available() else 0

logging.basicConfig(
    handlers=[
        logging.FileHandler(args.logfile, encoding="utf8"),
        logging.StreamHandler(),
    ],
    level=logging.INFO,
)
logging.info(args)


def fuse_incomplete_view_z(
    Z_v: List[Tensor],
    W_v: List[Tensor],
    output_shape: tuple,
):
    """
    将按照[A, U]排列的局部锚点图融合为全局锚点图（按原始样本顺序）
    """
    device = Z_v[0].device
    # 为了节省内存，分配一个原地加法的内存。
    numerator = torch.zeros(output_shape, device=device)
    dominator = torch.zeros(output_shape, device=device)
    for Z, W in zip(Z_v, W_v):
        numerator[W] += Z
        dominator[W] += 1

    # 除零错误处理。
    zero_places = dominator == 0.0
    assert torch.all(numerator[zero_places] == 0.0)
    dominator[dominator == 0] = 1  # 如果有0，说明分子也是零.

    Z_fused = numerator / dominator
    return Z_fused


class Preprocess(nn.Module):
    def __init__(self) -> None:
        super(Preprocess, self).__init__()

    def forward(self):
        data = PartialMultiviewDataset(
            datapath=P(args.datapath),
            view_ids=args.views,
            paired_rate=1 - args.per,
            normalize="minmax",
        )
        logging.info("Loaded dataset {}".format(data.name))
        A = data.X_paired_list
        U = data.X_single_list
        W = data.idx_all_list

        res = dict(
            data=data,
            viewNum=data.viewNum,
            mm=MaxMetrics(ACC=True, NMI=True, PUR=True, F1=True),
            A=convert_tensor(A, torch.float, args.device),
            U=convert_tensor(U, torch.float, args.device),
            W=convert_tensor(W, torch.float, args.device),
        )
        return res


class SparseSubspaceModel(nn.Module):

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.S = nn.Parameter(torch.empty(n, n))
        nn.init.normal_(self.S)
        self.M = 1 - torch.eye(n)
        self.M.requires_grad_(False)

    def forward(self, inputs: dict):
        logits = torch.exp(self.S) * self.M


class SparseSubspaceLoss(nn.Module):
       
    def forward(self, inputs: dict):
        X = inputs.get("X")        
        S = inputs.get("S")
        loss = F.mse_loss(S @ X, X)
        inputs['loss'] = loss
        return inputs

class InterViewConsensusModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: dict):
        pass


class IntraViewAnchorModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: dict):
        pass



class AnchorBasedSparseSubspaceModel(nn.Module):
    """
    补全模型，先预训练然后得到补全后的特征。
    """

    def __init__(self, d: int, in_channels: List[int]):
        super(AnchorBasedSparseSubspaceModel, self).__init__()

    def forward(self, inputs: dict):
        pass


class CompletionLoss(nn.Module):
    """
    补全损失函数。
    """

    def __init__(self, lamda: float):
        super(CompletionLoss, self).__init__()
        self.lamda = lamda

    def forward(self, inputs: dict):
        return inputs



class ClusteringPostProcess(nn.Module):
    def __init__(self) -> None:
        super(ClusteringPostProcess, self).__init__()

    def forward(self, inputs: dict):
        savedir: P = P(args.savedir)
        metrics_outfile = savedir.joinpath("metrics.json")
        mm: MaxMetrics = inputs["mm"]
        metrics = json.dumps(mm.report(current=False), indent=4)
        metrics_outfile.write_text(metrics)
        config = {
            key: str(val) if isinstance(val, P) else val
            for key, val in args.__dict__.items()
        }
        config = json.dumps(config, indent=4, ensure_ascii=False)
        config_outfile = savedir.joinpath("config.json")
        config_outfile.write_text(config)
        # TODO: lots of visualization

        H = inputs.get("H")  # pretrained latent
        H_bar = inputs.get("H_bar")  # finetuned latent
        S_bar = inputs.get("S_bar")  # Graph of completed features
        X = inputs.get("X")  # gt of input features X
        X_bar = inputs.get("X_bar")

        H = convert_numpy(H)
        H_bar = convert_numpy(H_bar)
        S_bar = convert_numpy(S_bar)
        X = convert_numpy(X)
        X_bar = convert_numpy(X_bar)

        post_scalers: List[MinMaxScaler] = inputs.get("post_scalers")
        for v in range(inputs["viewNum"]):
            X_bar[v] = post_scalers[v].inverse_transform(X_bar[v])

        return inputs


def main():
    preprocessor = Preprocess()
    inputs = preprocessor()
    data = inputs["data"]
    mm = inputs["mm"]
    completer_model = AnchorBasedSparseSubspaceModel(
        d=args.hidden_dims,
        in_channels=inputs["data"].view_dims,
    ).to(args.device)
    completer_loss = CompletionLoss(args.lamda)
    optim = torch.optim.Adam(completer_model.parameters(), lr=args.lr)
    logging.info("************* Begin train completer model **************")

    for epoch in range(args.epochs):
        completer_model.train()
        inputs = completer_model(inputs)
        inputs = completer_loss(inputs)
        loss = inputs["loss_completion"]["L"]
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (1 + epoch) % args.eval_epochs == 0:
            completer_model.eval()
            with torch.no_grad():
                inputs = completer_model(inputs)
            metrics = KMeans_Evaluate(inputs["H"], data)
            metrics["MSE"] = mse_missing_part(
                X_hat=inputs["X_hat"],
                X=inputs["X"],
                M=inputs["M"],
            )
            mm.update(**metrics)
            logging.info(f"epoch {epoch:04} {loss.item():.4f} {mm.report()}")

    logging.info("************* Begin postprocessing of Clustering **************")
    clustering_post = ClusteringPostProcess()
    inputs = clustering_post(inputs)


if __name__ == "__main__":
    main()
