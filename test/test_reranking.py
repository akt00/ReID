# import time

import torch

from src import reranking


def test_batch_euclidean():
    x = torch.randn((1, 2048))
    y = torch.randn((30, 2048))
    res = reranking.batch_euclidean(x, y, False)
    assert res.shape == (1, 30)


def test_batch_euclidean_batched():
    x = torch.randn((10, 2048))
    y = torch.randn((30, 2048))
    res = reranking.batch_euclidean(x, y, False)
    assert res.shape == (10, 30)


def test_batch_euclidean_normalize():
    x = torch.randn((1, 2048))
    y = torch.randn((30, 2048))
    res = reranking.batch_euclidean(x, y, True)
    res = res.mean().item()
    assert 0 <= res and res <= 2


class TestTrackID:
    """
    def test_update(self):
        i = reranking.TrackID()
        for j in range(2):
            i.update((j, j), torch.randn((1, 2048)))
        assert len(i.track_ids) == 2
        assert len(i.vector_store) == 2

    def test_update_id_duplicate(self):
        i = reranking.TrackID()
        for _ in range(2):
            i.update((1, 1), torch.randn((1, 2048)))
        assert len(i.track_ids) == 1
        assert len(i.vector_store) == 2

    def test_callable(self):
        i = reranking.TrackID()
        for _ in range(10):
            i.update((1, 1), torch.randn((1, 2048)))
        x = torch.randn((1, 2048))
        res = i(x, reduction=False)
        assert res.shape == (10,)

    def test_callable_reduction(self):
        i = reranking.TrackID(0)
        for _ in range(10):
            i.update((1, 1), torch.randn((1, 2048)))
        x = torch.randn((1, 2048))
        res = i(x, reduction=True)
        res = res.item()
        assert 0 <= res and res <= 2

    def test_duration(self):
        i = reranking.TrackID(0)
        time.sleep(1)
        assert i.duration() >= 1
    """
