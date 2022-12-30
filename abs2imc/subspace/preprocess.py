import logging
from pathlib import Path as P

from abs2imc.data import PartialMultiviewDataset
from abs2imc.utils.metrics import MaxMetrics
from abs2imc.utils.torch_utils import convert_tensor, nn, torch


class Preprocess(nn.Module):
    def __init__(self, args) -> None:
        super(Preprocess, self).__init__()
        self.args = args

    def forward(self):
        args = self.args
        data = PartialMultiviewDataset(
            datapath=P(args.datapath),
            view_ids=args.views,
            paired_rate=args.per,
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
            W=convert_tensor(W, torch.long, args.device),
            V=data.viewNum,
            n=data.sampleNum,
            n_a=data.pairedNum,
            n_u=[U[v].shape[0] for v in range(data.viewNum)],
        )
        return res
