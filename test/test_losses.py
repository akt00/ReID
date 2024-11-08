import torch
from torch import nn

from src import losses


def test_triplet_semi_hard_negative():
    e = torch.randn((800, 2048))
    model = nn.Sequential(nn.Linear(2048, 2048))
    out = model(e)
    t = torch.randint(0, 750, (800,)).long()
    res = losses.triplet_semi_hard_negative_mining(out, t)
    assert res.item() >= 0
