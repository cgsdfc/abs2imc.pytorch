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

from typing import List

from abs2imc.utils.torch_utils import F, Tensor, nn, torch, EPS_max


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
            Z_a.append(masked_softmax(self.Theta_a[v], self.M))
            Z_u.append(F.softmax(self.Theta_u[v], 1))
            Z_a_centroid += Z_a[v]

        Z_a_centroid = Z_a_centroid / self.V
        for v in range(self.V):
            Z.append(torch.cat((Z_a_centroid, Z_u[v])).detach())
        Z = fuse_incomplete_view_z(Z, W, (self.n, self.n_a))

        inputs.update(
            Z_a=Z_a,
            Z_u=Z_u,
            Z_a_centroid=Z_a_centroid,
            Z=Z,
        )
        return inputs
