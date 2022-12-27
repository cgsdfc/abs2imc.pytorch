import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import random
import numpy as np

from pathlib import Path as P
from typing import List

from abss_imc.data import PartialMultiviewDataset
# from abss_imc.model import *
# from abss_imc.utils import *
# from abss_imc.vis import *
from abss_imc.utils.metrics import Evaluate_Graph
from abss_imc.utils.torch_utils import EPS_max


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--per", type=float, default=0.5, help="paired example rate")
parser.add_argument("--lamda", type=float, default=0.1, help="lambda")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--datapath", type=str, default="./datasets/coil20_3view_1024.mat")
parser.add_argument("--views", type=str, default="0,1", help="view ids")
parser.add_argument("--savedir", type=str, default="./output/debug")
parser.add_argument("--logfile", type=str, default="./train.log")
parser.add_argument("--eval_epochs", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1, help="completer train epochs")


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
    Z: List[Tensor],
    W: List[Tensor],
    output_shape: tuple,
):
    """
    将按照[A, U]排列的局部锚点图融合为全局锚点图（按原始样本顺序）
    """
    device = Z[0].device
    # 为了节省内存，分配一个原地加法的内存。
    numerator = torch.zeros(output_shape, device=device)
    dominator = torch.zeros(output_shape, device=device)
    for v in range(len(Z)):
        numerator[W[v]] += Z[v]
        dominator[W[v]] += 1

    # 除零错误处理。
    zero_places = dominator == 0.0
    assert torch.all(numerator[zero_places] == 0.0)
    dominator[dominator == 0] = 1  # 如果有0，说明分子也是零.

    Z_fused = numerator / dominator
    return Z_fused


def masked_softmax(X: Tensor, M: Tensor):
    logits = torch.exp(X) * M
    normalization = EPS_max(logits.sum(1)).unsqueeze(1)
    masked_probas = logits / normalization
    return masked_probas


class Preprocess(nn.Module):
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
            mm=MaxMetrics(ACC=True, NMI=True, PUR=True, F1=True),
            A=convert_tensor(A, torch.float, args.device),
            U=convert_tensor(U, torch.float, args.device),
            W=convert_tensor(W, torch.float, args.device),
            V=data.viewNum,
            n=data.sampleNum,
            n_a=data.pairedNum,
            n_u=[U[v].shape[0] for v in range(data.viewNum)],
        )
        return res


class AnchorBasedSparseSubspaceModel(nn.Module):
    def __init__(self, n: int, n_a: int, V: int, n_u: List[int], **kwds):
        super(AnchorBasedSparseSubspaceModel, self).__init__()
        self.n = n
        self.n_a = n_a
        self.n_u = n_u
        self.V = V
        self.Theta_a = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_a, n_a)) for _ in range(V)]
        )
        self.Theta_u = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_u[i], n_a)) for i in range(V)]
        )
        # NOTE: Use Parameter for model.to(device)
        self.M = nn.Parameter(data=(1 - torch.eye(n_a)), requires_grad=False)

    def forward(self, inputs: dict):
        Z_a = []
        Z_u = []
        Z = []
        W = inputs.get("W")
        Z_a_centroid = torch.zeros_like(self.Theta_a[0])
        for v in range(self.V):
            Z_a.append(masked_softmax(self.Theta_a[v]))
            Z_u.append(F.softmax(self.Theta_u[v], 1))
            Z_a_centroid += Z_a[v]

        Z_a_centroid = Z_a_centroid / self.V
        for v in range(self.V):
            Z.append(torch.cat((Z_a_centroid, Z_u[v])).detach())
        Z_fused = fuse_incomplete_view_z(Z, W)

        inputs.update(
            Z_a=Z_a,
            Z_u=Z_u,
            Z_a_centroid=Z_a_centroid,
            Z_fused=Z_fused,
        )
        return inputs


class AnchorBasedSparseSubpaceLoss(nn.Module):
    def __init__(self, lamda: float, V: int):
        super(AnchorBasedSparseSubpaceLoss, self).__init__()
        self.lamda = lamda
        self.V = V

    def forward(self, inputs: dict):
        U = inputs.get("U")
        A = inputs.get("A")
        Z_a = inputs.get("Z_a")
        Z_u = inputs.get("Z_u")
        Z_a_centroid = inputs.get("Z_a_centroid")
        L_a1 = 0
        L_a2 = 0
        L_u = 0
        for v in range(self.V):
            L_a1 += F.mse_loss(Z_a[v] @ A[v], A[v])
            L_a2 += F.mse_loss(Z_a_centroid @ A[v], A[v])
            L_u += F.mse_loss(Z_u[v] @ A[v], U[v])

        loss = L_a1 + self.lamda * L_a2 + L_u
        inputs.update(
            loss=loss,
            L_a1=L_a1,
            L_a2=L_a2,
            L_u=L_u,
        )
        return inputs


class PostProcess(nn.Module):
    def __init__(self) -> None:
        super(PostProcess, self).__init__()

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

        return inputs


def main():
    preprocess = Preprocess()
    inputs = preprocess()
    data: PartialMultiviewDataset = inputs["data"]
    mm = inputs["mm"]
    subspace_model = AnchorBasedSparseSubspaceModel(**inputs).to(args.device)
    criterion = AnchorBasedSparseSubpaceLoss(args.lamda, data.viewNum)
    optim = torch.optim.Adam(subspace_model.parameters(), lr=args.lr)
    logging.info("************* Begin train subspace model **************")

    for epoch in range(args.epochs):
        subspace_model.train()
        inputs = subspace_model(inputs)
        inputs = criterion(inputs)
        loss = inputs["loss"]
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (1 + epoch) % args.eval_epochs == 0:
            subspace_model.eval()
            with torch.no_grad():
                inputs = subspace_model(inputs)
            metrics = Evaluate_Graph(data, Z=inputs["Z_fused"])
            mm.update(**metrics)
            logging.info(f"epoch {epoch:04} {loss.item():.4f} {mm.report()}")

    logging.info("************* Begin postprocessing of Clustering **************")
    postprocess = PostProcess()
    inputs = postprocess(inputs)


if __name__ == "__main__":
    main()
