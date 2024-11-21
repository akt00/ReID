import os

import pytest

from src import dataset


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestMarket1501:
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions."
    )
    def test_instance(self):
        _ = dataset.Market1501()
        assert True


class TestImageDataset:
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions."
    )
    def test_instance(self):
        _ = dataset.ImageDataset(dataset=dataset.Market1501().train)

    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions."
    )
    def test_iter(self):
        data = dataset.ImageDataset(dataset=dataset.Market1501().train)
        for img, pid in data:
            assert img.shape == (3, 224, 224)
            assert isinstance(pid.item(), int)
            break
