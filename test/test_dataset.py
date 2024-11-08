from src import dataset


class TestMarket1501:
    def test_instance(self):
        _ = dataset.Market1501()
        assert True


class TestImageDataset:
    def test_instance(self):
        _ = dataset.ImageDataset(dataset=dataset.Market1501().train)

    def test_iter(self):
        data = dataset.ImageDataset(dataset=dataset.Market1501().train)
        for img, pid in data:
            assert img.shape == (3, 224, 224)
            assert isinstance(pid.item(), int)
            break
