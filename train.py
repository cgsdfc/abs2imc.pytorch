import json
import os
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import argparse
import logging
import math
import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from anchor_based_subspace_imc.data import *
from anchor_based_subspace_imc.model import *
from anchor_based_subspace_imc.utils import *
from anchor_based_subspace_imc.vis import *

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
parser.add_argument(
    "--mask_kind",
    type=str,
    default="general",
    choices=[
        "general",
        "partial",
        "weaker",
    ],
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


class Preprocess(nn.Module):
    def __init__(self) -> None:
        super(Preprocess, self).__init__()

    def forward(self):
        data = MultiviewDataset(
            datapath=P(args.datapath),
            view_ids=args.views,
        )
        logging.info("Loaded dataset {}".format(data.name))

        # 缺失指示矩阵 M
        M = make_mask(
            paired_rate=args.per,
            sampleNum=data.sampleNum,
            viewNum=data.viewNum,
            kind=args.mask_kind,
        )

        # 每个视角的实际存在样本
        X_tilde = [data.X[v][M[:, v]] for v in range(data.viewNum)]
        Scaler = [MinMaxScaler() for _ in range(data.viewNum)]
        for v in range(data.viewNum):
            X_tilde[v] = Scaler[v].fit_transform(X_tilde[v])
        X_tilde = convert_tensor(X_tilde, torch.float, args.device)

        # 未归一化的高斯核
        S_tilde = [
            calculate_optimized_p_cond(x_tilde, math.log2(args.ppl), dev=args.device)
            for x_tilde in X_tilde
        ]

        # 原始数据联合概率分布
        P_tilde = [make_joint(s_tilde) for s_tilde in S_tilde]

        res = dict(
            data=data,
            viewNum=data.viewNum,
            pre_scalers=Scaler,
            M=convert_tensor(M, torch.bool, args.device),
            S_tilde=S_tilde,
            P_tilde=P_tilde,
            X_tilde=X_tilde,
            X=convert_tensor(data.X, torch.float, args.device),
            mm=MaxMetrics(ACC=True, NMI=True, PUR=True, F1=True),
        )
        return res


class CompleterModel(nn.Module):
    """
    补全模型，先预训练然后得到补全后的特征。
    """

    def __init__(self, d: int, in_channels: List[int]):
        super(CompleterModel, self).__init__()
        self.Encoder = nn.ModuleList()
        self.Decoder = nn.ModuleList()
        self.d = d
        for v, in_channel in enumerate(in_channels):
            if args.use_mlp:
                encoder = NeuralMapper(dim_input=in_channel, dim_emb=d)
            else:
                encoder = GCN_Encoder_SDIMC(view_dim=in_channel, clusterNum=d)
            decoder = Imputer(dim_output=in_channel, dim_emb=d)
            self.Encoder.append(encoder)
            self.Decoder.append(decoder)

    def forward(self, inputs: dict):
        X_tilde: List[Tensor] = inputs.get("X_tilde")
        M: Tensor = inputs.get("M")
        S_tilde: List[Tensor] = inputs.get("S_tilde")

        H_tilde = []
        viewNum = len(X_tilde)
        sampleNum = M.shape[0]
        # Encoding
        for v in range(viewNum):
            if args.use_mlp:
                h_tilde = self.Encoder[v](X_tilde[v])
            else:
                h_tilde = self.Encoder[v](X_tilde[v], S_tilde[v])
            H_tilde.append(h_tilde)
        # Fusion
        H = torch.zeros(sampleNum, self.d).to(args.device)
        for v in range(viewNum):
            H[M[:, v]] += H_tilde[v]
        # Decoding
        X_hat = []
        for v in range(viewNum):
            x_hat = self.Decoder[v](H)
            X_hat.append(x_hat)

        inputs["H"] = H
        inputs["X_hat"] = X_hat
        return inputs


class CompletionLoss(nn.Module):
    """
    补全损失函数。
    """

    def __init__(self, lamda: float):
        super(CompletionLoss, self).__init__()
        self.lamda = lamda

    def forward(self, inputs: dict):
        P_tilde: List[Tensor] = inputs.get("P_tilde")
        H: Tensor = inputs.get("H")
        M: Tensor = inputs.get("M")
        X: List[Tensor] = inputs.get("X")
        X_hat: List[Tensor] = inputs.get("X_hat")
        # Loss representation
        H_bar = []
        Q_bar = []
        sampleNum, viewNum = M.shape
        L_tsne = 0
        for v in range(viewNum):
            h_bar = H[M[:, v]]
            q_bar = get_q_joint(h_bar)
            H_bar.append(h_bar)
            Q_bar.append(q_bar)
            L_tsne += loss_function(p_joint=P_tilde[v], q_joint=q_bar)
        # Loss completion
        L_mse = 0
        for v in range(viewNum):
            pred = X_hat[v][M[:, v]]
            gt = X[v][M[:, v]]
            L_mse += F.mse_loss(pred, gt)
        L = L_tsne + self.lamda * L_mse
        inputs.update(loss_completion=dict(L=L, L_tsne=L_tsne, L_mse=L_mse))
        return inputs


class CompletionPostProcess(nn.Module):
    """
    补全模型训练完毕后，利用它补全特征，重新归一化，获取初始类簇给聚类模型初始化。
    """

    def __init__(self, pretrained_model: CompleterModel) -> None:
        super(CompletionPostProcess, self).__init__()
        model_path = P(args.savedir).joinpath("completer-model.pth")
        # torch.save(pretrained_model.state_dict(), model_path.open("wb"))
        self.completer_model = pretrained_model
        self.completer_model_path = model_path

    def forward(self, inputs: dict):
        self.completer_model.eval()
        with torch.no_grad():
            inputs = self.completer_model(inputs)

        X_hat = inputs.get("X_hat")
        X = inputs.get("X")
        M = inputs.get("M")
        M = convert_numpy(M)
        H = inputs.get("H")
        pre_scalers: List[MinMaxScaler] = inputs.get("pre_scalers")
        post_scalers = [MinMaxScaler() for _ in range(inputs["viewNum"])]
        X_bar = []
        # Completion
        for v in range(inputs["viewNum"]):
            x_hat = X_hat[v]
            x_hat = convert_numpy(x_hat)
            x_hat = pre_scalers[v].inverse_transform(x_hat)
            x = X[v]
            x = convert_numpy(x)
            x_bar = np.zeros_like(x)
            x_bar[M[:, v]] = x_hat[M[:, v]]
            x_bar[~M[:, v]] = x[~M[:, v]]
            x_bar = post_scalers[v].fit_transform(x_bar)
            x_bar = convert_tensor(x_bar, torch.float, args.device)
            X_bar.append(x_bar)
        # Initial Clustering
        data = inputs["data"]
        metrics, centroid, ypred = KMeans_Evaluate(H, data, return_centroid=True)
        mm = inputs["mm"]
        mm.update(**metrics)
        logging.info("After completion pretraining {}".format(mm.report()))
        # Build Complete Graph
        S_bar = [
            calculate_optimized_p_cond(x_bar, math.log2(args.ppl), dev=args.device)
            for x_bar in X_bar
        ]
        P_bar = [make_joint(s_bar) for s_bar in S_bar]
        # Save Output Variables
        inputs["centroid"] = centroid
        inputs["ypred"] = ypred
        inputs["S_bar"] = S_bar
        inputs["X_bar"] = X_bar
        inputs["P_bar"] = P_bar
        inputs["H"] = H
        inputs["post_scalers"] = post_scalers
        inputs["completer_model"] = self.completer_model
        inputs["completer_model_path"] = self.completer_model_path
        return inputs


class ClusteringModel(nn.Module):
    def __init__(self, inputs: dict) -> None:
        super(ClusteringModel, self).__init__()
        self.Encoder = inputs["completer_model"].Encoder
        self.Mu = nn.Parameter(inputs["centroid"])

    def forward(self, inputs: dict):
        X_bar: List[Tensor] = inputs.get("X_bar")
        S_bar: List[Tensor] = inputs.get("S_bar")
        H_bar = torch.zeros_like(inputs["H"]).to(args.device)
        # Encoding
        for v in range(inputs.get("viewNum")):
            if args.use_mlp:
                h = self.Encoder[v](X_bar[v])
            else:
                h = self.Encoder[v](X_bar[v], S_bar[v])
            H_bar += h
        # Soft clustering
        Q_cluster = get_q_cluster(H_bar, self.Mu)
        inputs["H_bar"] = H_bar
        inputs["Q_cluster"] = Q_cluster
        return inputs


class ClusteringLoss(nn.Module):
    def __init__(self) -> None:
        super(ClusteringLoss, self).__init__()
        self.beta = args.beta

    def forward(self, inputs: dict):
        Q_cluster: Tensor = inputs.get("Q_cluster")
        P_cluster: Tensor = inputs.get("P_cluster")
        P_bar: List[Tensor] = inputs.get("P_bar")
        H_bar = inputs.get("H_bar")
        L_cluster = kl_clustering_loss(q=Q_cluster, p=P_cluster)
        Q_bar = get_q_joint(H_bar)
        L_tsne = 0
        for v in range(inputs["viewNum"]):
            L_tsne += loss_function(p_joint=P_bar[v], q_joint=Q_bar)

        L = L_tsne + self.beta * L_cluster
        inputs["loss_cluster"] = dict(L=L, L_tsne=L_tsne, L_cluster=L_cluster)
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


def main():
    preprocessor = Preprocess()
    inputs = preprocessor()
    data = inputs["data"]
    mm = inputs["mm"]
    completer_model = CompleterModel(
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

    logging.info("************* Begin postprocessing of Completion **************")

    completion_post = CompletionPostProcess(completer_model)
    inputs = completion_post(inputs)
    clustering_model = ClusteringModel(inputs).to(args.device)
    clustering_loss = ClusteringLoss()
    optim = torch.optim.Adam(clustering_model.parameters(), lr=args.lr)

    # TODO: 按照KL训练的算法进行。
    P_cluster = None
    loss = 0
    ypred = inputs["ypred"]
    logging.info("************* Begin train clustering model **************")
    while True:
        clustering_model.train()
        if epoch % args.cluster_update_epoch == 0:
            # 注意，这里epoch不要加1，因为0，即初始化。
            clustering_model.eval()
            inputs: dict = clustering_model(inputs)
            Q_cluster = inputs["Q_cluster"]
            P_cluster = target_distribution(Q_cluster)
            inputs.update(P_cluster=P_cluster)
            ypred_Q = torch.argmax(Q_cluster, 1)
            metrics = get_all_metrics(data.Y, convert_numpy(ypred_Q))
            mm.update(**metrics)
            print(f"epoch {epoch:05} {loss:.4f} {mm.report()}")
            diff = torch.abs(ypred - ypred_Q).int()
            diff = torch.count_nonzero(diff) / data.sampleNum
            if diff < args.stop_thresh:
                logging.info(
                    "diff={}, stop-thresh={}, STOP".format(diff, args.stop_thresh)
                )
                break

        inputs = clustering_model(inputs)
        inputs = clustering_loss(inputs)
        loss = inputs["loss_cluster"]["L"]
        optim.zero_grad()
        loss.backward()
        optim.step()

    logging.info("************* Begin postprocessing of Clustering **************")
    clustering_post = ClusteringPostProcess()
    inputs = clustering_post(inputs)


if __name__ == "__main__":
    main()
