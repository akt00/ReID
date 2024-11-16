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
        """returns the number of embeddings"""
        return len(self.vector_store)


class ReRanker:
    def __init__(self, topk: int = 3, device: torch.device = torch.device("cuda")):
        self.topk = topk
        self.device = device
        self.vector_store: list[Tensor] = []
        self.targets: list[Tensor] = []

    def append(self, x: Tensor, y: Tensor | list):
        """stores embeddings
        Args:
            x: embedding tensor with shape (batch, n, ...)
            y: embedding labels
        """
        assert len(x.shape) > 1
        x = x.reshape(x.size(0), -1)
        self.vector_store += list(x)
        if isinstance(y, Tensor):
            self.targets += y.tolist()
        else:
            self.targets += y

    def predict(self, x: Tensor) -> Tensor:
        """predicts the top-k class labels
        Args:
            x: batched query embeddings, (batch, N, ...)
        Returns:
            the predicted class labels in (batch, top-k)
        """
        x = x.reshape(x.size(0), -1)
        x = x.to(device=self.device)
        # n, dim = x.shape
        vs = torch.stack(self.vector_store)
        vs = vs.to(device=self.device)
        # m, _ = vs.shape
        dists = batched_euclidean(x=x, y=vs, cosine=True)
        topk_preds = dists.topk(self.topk, largest=False, dim=-1)

        indices = topk_preds.indices
        targets = torch.tensor(self.targets, device=self.device).long()

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
        x = x.to(device=self.device)
        y = y.to(device=self.device)

        preds = self.predict(x)

        if knn:
            preds = preds.mode().values
            return (preds == y).long()
        else:
            target = y.unsqueeze(-1).expand_as(preds)
            return (preds == target).any(dim=-1).long()

    def __len__(self) -> int:
        """returns the number of embeddings"""
        assert len(self.vector_store) == len(self.targets)
        return len(self.vector_store)


class ClusterReRanker:
    def __init__(
        self, clusters: dict | None = None, device: torch.device = torch.device("cuda")
    ):
        self.device = device
        self.clusters: list[TrackID] = []
        self.ttls: list[int] = []

        if clusters is not None:
            for k, v in clusters.items():
                track = TrackID(k)
                track.update(v)
                self._append_new_cluster(track)

    def predict(self, x: Tensor) -> Tensor:
        x = x.to(device=self.device)
        dists = []

        for c in self.clusters:
            preds: Tensor = c(x)
            dists.append(preds)

        dists = torch.stack(dists, dim=-1)
        indices = dists.min(dim=-1).indices
        preds = torch.tensor(
            [self.clusters[i].id for i in indices], dtype=torch.long, device=self.device
        )

        return preds

    def evalute(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        preds = self.predict(x)
        return (preds == y).long()

    def _append_new_cluster(self, track: TrackID):
        self.clusters.append(track)
        self.ttls.append(time.time())

    def __len__(self) -> int:
        assert len(self.clusters) == len(self.ttls)
        return len(self.clusters)
