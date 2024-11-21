import torch

from src import reranking


def test_batch_euclidean():
    x = torch.randn((1, 2048))
    y = torch.randn((30, 2048))
    res = reranking.batched_euclidean(x, y, False)
    assert res.shape == (1, 30)


def test_batch_euclidean_batched():
    x = torch.randn((10, 2048))
    y = torch.randn((30, 2048))
    res = reranking.batched_euclidean(x, y, False)
    assert res.shape == (10, 30)


def test_batch_euclidean_normalize():
    x = torch.randn((1, 2048))
    y = torch.randn((30, 2048))
    res = reranking.batched_euclidean(x, y, True)
    res = res.mean().item()
    assert 0 <= res and res <= 2


class TestTrackID:
    def test_update(self):
        track = reranking.TrackID(0)
        for _ in range(2):
            track.update(torch.randn((1, 2048)))
        assert track.id == 0
        assert len(track.vector_store) == 2

    def test_callable_all_points(self):
        track = reranking.TrackID(0)
        for _ in range(10):
            track.update(torch.randn((1, 2048)))
        x = torch.randn((10, 2048))
        res = track(x, kmeans=False)
        assert res.shape == (10,)

    def test_callable_kmeans(self):
        track = reranking.TrackID(0)
        for _ in range(10):
            track.update(torch.randn((1, 2048)))
        x = torch.randn((10, 2048))
        res = track(x, kmeans=True)
        assert res.shape == (10,)


class TesteReRanker:
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_append(self):
        ranker = reranking.ReRanker(device=self.DEVICE)
        x = torch.randn((16, 128), device=self.DEVICE)
        y = torch.randint(0, 5, (16,), device=self.DEVICE)
        ranker.append(x=x, y=y)
        assert len(ranker) == 16

    def test_predict(self):
        ranker = reranking.ReRanker(device=self.DEVICE)
        x = torch.randn((16, 128), device=self.DEVICE)
        y = torch.randint(0, 5, (16,), device=self.DEVICE)
        ranker.append(x=x, y=y)
        preds = ranker.predict(x=x)
        assert preds.shape == (16, 5)

    def test_evaluate(self):
        ranker = reranking.ReRanker(device=self.DEVICE)
        x = torch.randn((16, 128), device=self.DEVICE)
        y = torch.randint(0, 5, (16,), device=self.DEVICE)
        ranker.append(x=x, y=y)
        preds = ranker.evaluate(x=x, y=y)
        assert preds.shape == (16,)


class TestClusterReRanker:
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_predict(self):
        clusters = {}
        for i in range(5):
            clusters.update({i: torch.randn((4, 128), device=self.DEVICE)})
        ranker = reranking.ClusterReRanker(clusters=clusters, device=self.DEVICE)
        preds = ranker.predict(x=torch.randn((4, 128), device=self.DEVICE))
        assert preds.shape == (4,)

    def test_evaluate(self):
        clusters = {}
        for i in range(5):
            clusters.update({i: torch.randn((4, 128), device=self.DEVICE)})
        ranker = reranking.ClusterReRanker(clusters=clusters, device=self.DEVICE)
        preds = ranker.evalute(
            x=torch.randn((4, 128), device=self.DEVICE),
            y=torch.randint(0, 5, (4,), device=self.DEVICE),
        )
        assert preds.shape == (4,)

    def test_append(self):
        clusters = {}
        for i in range(5):
            clusters.update({i: torch.randn((4, 128), device=self.DEVICE)})
        ranker = reranking.ClusterReRanker(clusters=clusters, device=self.DEVICE)
        ranker.append(x=torch.randn((4, 128), device=self.DEVICE), index=2)
        assert len(ranker.clusters[2]) == 8
