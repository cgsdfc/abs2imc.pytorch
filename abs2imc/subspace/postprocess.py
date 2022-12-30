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

import json
from pathlib import Path as P

from abs2imc.utils.metrics import Evaluate_Graph, MaxMetrics
from abs2imc.utils.torch_utils import convert_numpy, nn
from abs2imc.vis.plot_utils import pairwise_distances, plt, sns


class PostProcess(nn.Module):
    def __init__(self, args) -> None:
        super(PostProcess, self).__init__()
        self.args = args

    def forward(self, inputs: dict):
        args = self.args
        savedir: P = P(args.savedir)

        metrics_outfile = savedir.joinpath("metrics.json")
        mm: MaxMetrics = inputs["mm"]
        metrics = json.dumps(mm.report(current=False), indent=4)
        print('Best metrics', metrics)
        metrics_outfile.write_text(metrics)

        config = {
            key: str(val) if isinstance(val, P) else val
            for key, val in args.__dict__.items()
        }
        config = json.dumps(config, indent=4, ensure_ascii=False)
        config_outfile = savedir.joinpath("config.json")
        config_outfile.write_text(config)

        Z = inputs.get("Z")
        data = inputs.get("data")
        P_ = Evaluate_Graph(data, Z=Z, return_spectral_embedding=True)
        D = pairwise_distances(P_)
        Z = convert_numpy(Z)

        sns.heatmap(D, cmap='winter')
        plt.title('Block diagonal structure of spectral embeddings $P$')
        plt.savefig(str(savedir.joinpath("P-blockdiag.jpg")))
        plt.close()

        sns.heatmap(Z, cmap='winter')
        plt.title('Complete consensus sparse anchor graph $Z$')
        plt.savefig(str(savedir.joinpath("Z.jpg")))
        plt.close()

        return inputs
