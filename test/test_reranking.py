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
