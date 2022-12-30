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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import random
from pathlib import Path as P

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--per", type=float, default=0.3, help="paired example rate")
parser.add_argument("--lamda", type=float, default=0.1, help="lambda")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--datapath", type=str, default="./dataset/handwrittenRnSp.mat")
parser.add_argument("--views", type=str, default=None, help="view ids")
parser.add_argument("--savedir", type=str, default="./output/debug")
parser.add_argument("--logfile", type=str, default="./train.log")
parser.add_argument("--eval_epochs", type=int, default=10)
parser.add_argument("--epochs", type=int, default=200, help="train epochs")


args = parser.parse_args()
if args.views is not None:
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

__all__ = ["args"]
