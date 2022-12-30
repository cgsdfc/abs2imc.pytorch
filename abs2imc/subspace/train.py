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

import torch

from .config import args
from .loss import AnchorBasedSparseSubpaceLoss
from .model import AnchorBasedSparseSubspaceModel
from .postprocess import PostProcess
from .preprocess import Preprocess
from abs2imc.utils.metrics import Evaluate_Graph
from abs2imc.data.dataset import PartialMultiviewDataset


def train_abs2imc():
    preprocess = Preprocess(args)
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
        loss = inputs["loss"]["L"]
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (1 + epoch) % args.eval_epochs == 0:
            subspace_model.eval()
            with torch.no_grad():
                inputs = subspace_model(inputs)
            metrics = Evaluate_Graph(data, Z=inputs["Z"])
            mm.update(**metrics)
            logging.info(f"epoch {epoch:04} {loss.item():.4f} {mm.report()}")

    logging.info("************* Begin postprocessing of Clustering **************")
    postprocess = PostProcess(args)
    inputs = postprocess(inputs)
