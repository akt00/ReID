import torch

from src.models import HorizontalMaxPool2d, AlignedResNet50


class TestOnGPU:
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_hmp2d(self):
        x = torch.randn((8, 3, 7, 7)).to(device=self.DEVICE)
        model = HorizontalMaxPool2d().to(device=self.DEVICE)
        assert (8, 3, 7, 1) == model(x).shape

    def test_aligned_resnet50_train(self):
        x = torch.randn((8, 3, 224, 224)).to(device=self.DEVICE)
        model = AlignedResNet50()
        model.train().to(device=self.DEVICE)
        ge, le = model(x)
        assert (8, 2048, 1) == ge.shape
        assert (8, 128, 7) == le.shape

    def test_aligned_resnet50_eval(self):
        x = torch.randn((8, 3, 224, 224)).to(device=self.DEVICE)
        model = AlignedResNet50()
        model.eval().to(device=self.DEVICE)
        assert (8, 2048, 1) == model(x).shape
