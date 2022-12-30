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
