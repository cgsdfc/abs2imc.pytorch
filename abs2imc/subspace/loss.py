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

from abs2imc.utils.torch_utils import F, nn


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
        inputs["loss"] = dict(
            L_a1=L_a1,
            L_a2=L_a2,
            L_u=L_u,
            L=loss,
        )
        return inputs
