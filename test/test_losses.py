import torch

from src import losses


def test_triplet_semi_hard_negative():
    e = torch.randn((256, 2048))
    y = torch.randint(0, 100, (256,)).long()
    loss, count = losses.triplet_semi_hard_negative_mining(e, y)
    assert loss.item() >= 0
    assert count >= 0


def test_shortest_path():
    e1 = torch.randn((8, 7, 128))
    e2 = torch.randn((8, 7, 128))
    dist_mat = torch.cdist(e1, e2)
    sp = losses.shortest_path(dist_mat.permute(1, 2, 0))
    assert sp.shape == (8,)


def test_aligned_triplet_semi_hard_negative():
    e = torch.randn((8, 2048))
    le = torch.randn((8, 128, 7))
    y = torch.randint(0, 10, (8,)).long()
    loss, count = losses.aligned_triplet_semi_hard_negative_mining(e, le, y)
    assert loss.item() >= 0
    assert count >= 0
