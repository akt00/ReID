import torch
from torch import nn

from src import losses


def test_triplet_semi_hard_negative():
    e = torch.randn((256, 2048))
    model = nn.Sequential(nn.Linear(2048, 2048))
    x = model(e)
    y = torch.randint(0, 100, (256,)).long()
    loss, count = losses.triplet_semi_hard_negative_mining(x, y)
    assert loss.item() >= 0
    assert count > 0
