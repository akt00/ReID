from src import dataset


class TestMarket1501:
    def test_instance(self):
        _ = dataset.Market1501()
        assert True
