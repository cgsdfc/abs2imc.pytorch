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
