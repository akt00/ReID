import time

import torch
from torch import Tensor


def batched_euclidean(x: Tensor, y: Tensor, cosine: bool = True) -> Tensor:
    """computes cartesian batch L2 norms
    Args:
        x: query tensor of shape (n, dim, ...)
        y: gallery tensor of shape (m, dim, ...)
        cosine: rescales to unit vector
    Returns:
        dists: L2 norms in the last feature dim, (n, m, ...)
    """
    if cosine:
        x = x / (torch.linalg.norm(x, ord=2, dim=-1, keepdim=True) + 1e-10)
        y = y / (torch.linalg.norm(y, ord=2, dim=-1, keepdim=True) + 1e-10)

    n, *dim = x.shape
    m, *dim = y.shape
    x = x[:, None, :].expand(n, m, *dim)
    y = y[None, :, :].expand(n, m, *dim)
    dists = torch.linalg.norm(x - y, ord=2, dim=-1, keepdim=False)

    return dists


class TrackID:
    def __init__(self, track_id: int):
        self.id = track_id
        # store embeddings
        self.vector_store: list[Tensor] = []
        self.creation_time = time.time()

    def update(self, embs: Tensor):
        """add embedddings to the vector store
        Args:
            embs: embeddings with shape (batch, dim, ...)
        """
        assert len(embs.shape) > 1
        embs = embs.reshape(embs.size(0), -1)
        self.vector_store += list(embs)

    def duration(self) -> int:
        """
        Returns:
            dur: object duration in seconds
        """
        dur = int(time.time() - self.creation_time)
        return dur

    def __call__(self, x: Tensor, kmeans: bool = True) -> Tensor:
        """coomputes the distances between x and vector store
        Args:
            x: embeddings with shape (batch, dim)
            kmeans: either distance to the kmeans cluster or average distance to each cluster point
        Returns:
            dists: distances with (batch,)
        """
        gallery = torch.stack(tensors=self.vector_store, dim=0)
        if kmeans:
            cp = gallery.mean(dim=0)
            # (dim,) -> (1, dim)
            cp = cp.unsqueeze(dim=0)
            cp = cp.expand(x.size(0), cp.size(-1))
            cp = cp / (torch.linalg.norm(cp, ord=2, dim=-1, keepdim=True) + 1e-10)
            x = x / (torch.linalg.norm(x, ord=2, dim=-1, keepdim=True) + 1e-10)
            dists: Tensor = torch.linalg.norm(x - cp, ord=2, dim=-1, keepdim=False)
            return dists
        else:
            dists = batched_euclidean(x, gallery)
            return dists.mean(dim=-1)

    def __len__(self) -> int:
        """returns the number of embeddings"""
        return len(self.vector_store)


class ReRanker:
    def __init__(self, device: torch.device = torch.device("cuda")):
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

    def predict(self, x: Tensor, topk: int = 5) -> Tensor:
        """predicts the top-k class labels
        Args:
            x: batched query embeddings, (batch, N, ...)
        Returns:
            the predicted class labels in (batch, top-k)
        """
        assert len(self) > 0
        x = x.reshape(x.size(0), -1)
        x = x.to(device=self.device)
        # n, dim = x.shape
        vs = torch.stack(self.vector_store)
        vs = vs.to(device=self.device)
        # m, _ = vs.shape
        dists = batched_euclidean(x=x, y=vs, cosine=True)
        topk_preds = dists.topk(topk, largest=False, dim=-1)

        indices = topk_preds.indices
        targets = torch.tensor(self.targets, device=self.device).long()

        return targets[indices]

    def evaluate(
        self, x: Tensor, y: Tensor, topk: int = 5, knn: bool = False
    ) -> Tensor:
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

        preds = self.predict(x=x, topk=topk)

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
        self,
        clusters: dict[int, Tensor] | None = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.clusters: list[TrackID] = []
        self.ttls: list[int] = []

        if clusters is not None:
            for k, v in clusters.items():
                track = TrackID(k)
                track.update(v)
                self._append_new_cluster(track)

    def append(self, x: Tensor, index: int):
        """appends embeddings to an existing cluster

        Args:
            x: embeddings with shape (batch, dim, ...)
            index: the cluster index obtained from the predict method
        """
        self.clusters[index].update(embs=x)

    def predict(self, x: Tensor, kmeans: bool = True) -> Tensor:
        """model inference on input tensor x
        Args:
            x: input query embeddings with shape (batch, dim)
            kmeans: use kmeans center for distance metric
        Returns:
            preds: predicted indices with shape (batch,)
        """
        x = x.to(device=self.device)
        dists = []

        for c in self.clusters:
            preds: Tensor = c(x, kmeans=kmeans)
            dists.append(preds)

        dists = torch.stack(dists, dim=-1)
        indices = dists.min(dim=-1).indices
        preds = torch.tensor(
            [self.clusters[i].id for i in indices], dtype=torch.long, device=self.device
        )

        return preds

    def evalute(self, x: Tensor, y: Tensor, kmeans: bool = True) -> Tensor:
        """evaluates the predicted IDs
        Args:
            x: query embeddings with shape (batch, dim)
            y: ground truth labels with shape (batch,)
            kmeans: use kmeans center for distance metric
        Returns:
            preds: evaluation results with shape (batch,) where correct predictions are 1, 0 otherwise
        """
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        preds = self.predict(x, kmeans=kmeans)
        return (preds == y).long()

    def _append_new_cluster(self, track: TrackID):
        """appends a new cluster"""
        if track.id in [[c.id for c in self.clusters]]:
            raise ValueError(f"The cluster id already exists: {track.id}")

        self.clusters.append(track)
        self.ttls.append(time.time())

    def __len__(self) -> int:
        """returns the number of clusters"""
        assert len(self.clusters) == len(self.ttls)
        return len(self.clusters)
