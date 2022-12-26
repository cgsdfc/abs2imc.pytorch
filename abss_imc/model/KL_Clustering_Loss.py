import torch
import torch.nn.functional as F


def get_q_cluster(z, mu, alpha=1):
    """
    z: embeddings; mu: centroids
    q_{iu} = P(cluster(i)==u|i)
    表示给定i，i的类簇是u的概率。
    """
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - mu, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def kl_clustering_loss(q, p):
    # 注意，这个kl和tsne的kl是不同的。
    kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
    return kl_loss
