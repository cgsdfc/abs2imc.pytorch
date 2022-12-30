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
