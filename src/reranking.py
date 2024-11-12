import time

import torch
from torch import Tensor


def batched_euclidean(x: Tensor, y: Tensor, cosine: bool = True) -> Tensor:
    """
    Args:
        x: query tensor of shape (n, dim)
        y: gallery tensor of shape (m, dim)
        cosine: rescales to unit vector
    Returns:
        dists: L2 norms in the feature dim, (n, m)
    """
    if cosine:
        x = x / (torch.linalg.norm(x, ord=2, dim=-1, keepdim=True) + 1e-10)
        y = y / (torch.linalg.norm(y, ord=2, dim=-1, keepdim=True) + 1e-10)

    n, dim = x.shape
    m, _ = y.shape
    x = x[:, None, :].expand(n, m, dim)
    y = y[None, :, :].expand(n, m, dim)
    dists = torch.linalg.norm(x - y, ord=2, dim=-1, keepdim=False)

    return dists


class TrackID:
    def __init__(self, track_id: int):
        self.id = track_id
        self.creation_time = time.time()
        self.vector_store: list[Tensor] = []
        # more features will be added...
        # self.track_ids: deque[tuple] = deque()

    def update(self, emb: Tensor):
        assert len(emb.shape) > 1
        emb = emb.reshape(emb.size(0), -1)
        self.vector_store += list(emb)

    def duration(self) -> int:
        """
        Returns:
            dur: object duration in seconds
        """
        dur = int(time.time() - self.creation_time)
        return dur

    def __call__(self, x: Tensor, reduction: bool = True) -> Tensor:
        gallery = torch.stack(tensors=self.vector_store, dim=0)
        dists = batched_euclidean(x, gallery)
        if reduction:
            return dists.mean(dim=-1)
        else:
            return dists

    def __len__(self) -> int:
        return len(self.vector_store)


class ReRanker:
    def __init__(self, topk: int = 5):
        self.topk = topk
        self.vector_store = []
        self.targets = []

    def append(self, x: Tensor, y: Tensor):
        """stores embeddings
        Args:
            x: embedding tensor with shape (batch, n, ...)
            y: embedding labels
        """
        assert len(x.shape) > 1
        x = x.reshape(x.size(0), -1)
        self.vector_store += list(x)
        self.targets += y.tolist()

    def predict(self, x: Tensor) -> Tensor:
        """predicts the top-k class labels
        Args:
            x: batched query embeddings, (batch, N, ...)
        Returns:
            the predicted class labels in (batch, top-k)
        """
        x = x.reshape(x.size(0), -1)
        n, dim = x.shape
        vs = torch.stack(self.vector_store)
        m, _ = vs.shape

        x = x[:, None, :].expand(n, m, dim)
        vs = vs[None, :, :].expand(n, m, dim)

        l2_norms: Tensor = torch.linalg.norm((x - vs), ord=2, dim=-1, keepdim=False)
        topk_preds = l2_norms.topk(self.topk, dim=-1)
        indices = topk_preds.indices

        targets = torch.tensor(self.targets).long()

        return targets[indices]

    def evaluate(self, x: Tensor, y: Tensor, knn: bool = False) -> Tensor:
        """evalute the class predictions
        Args:
            x: batched query embeddings, (batch, N, ...)
            y: ground truth labels in (batch,)
            knn: use KNN instead of top-k
        Returns:
            1d tensor containing either 0 or 1 indicating the correct prediction
        """
        pred_labels = self.predict(x)
        # print(pred_labels)
        if knn:
            pred_labels = pred_labels.mode().values
            return (y == pred_labels).long()
        else:
            target = y.unsqueeze(-1).expand_as(pred_labels)
            return (target == pred_labels).any(-1).long()


class ClusterReRanker:
    def __init__(self, clusters: dict | None = None):
        self.clusters = []
        self.ttls = []
        if clusters is not None:
            for k, v in clusters.items():
                track = TrackID(int(k))
                track.update(v)
                self._append_new_cluster(track)

    def predict(self, x: Tensor) -> Tensor:
        dists = []
        for c in clusters:
            ...

    def _append_new_cluster(self, track: TrackID):
        self.clusters.append(track)
        self.ttls.append(time.time())

    def __len__(self) -> int:
        assert len(self.clusters) == len(self.ttls)
        return len(self.clusters)


if __name__ == "__main__":
    """
    x = torch.randn((1000, 2048))
    y = torch.randint(0, 5, (1000,))
    print(y.shape)
    ranker = ReRanker()
    ranker.append(x, y)

    inputs = torch.randn(4, 2048)
    tgt = torch.randint(0, 5, (4,))
    preds = ranker.evaluate(inputs, tgt, False)
    print(tgt)
    print(preds)
    """
    clusters = {}
    for i in range(10):
        clusters.update({i: torch.randn((10, 2048))})
    ranker = ClusterReRanker(clusters)
    print(ranker.clusters)
    for i in ranker.clusters:
        x = torch.randn((1, 2048))
        res = i(x)
        print(res)
