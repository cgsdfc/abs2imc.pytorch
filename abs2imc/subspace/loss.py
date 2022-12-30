from abs2imc.utils.torch_utils import (EPS_max, F, Tensor, convert_numpy,
                                       convert_tensor, nn, torch)


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
