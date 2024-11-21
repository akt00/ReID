import torch

from src.models import HorizontalMaxPool2d, AlignedResNet50


class TestOnGPU:
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_hmp2d(self):
        x = torch.randn((8, 3, 7, 7)).to(device=self.DEVICE)
        model = HorizontalMaxPool2d().to(device=self.DEVICE)
        assert model(x).shape == (8, 3, 7, 1)

    def test_aligned_resnet50_train(self):
        x = torch.randn((8, 3, 224, 224)).to(device=self.DEVICE)
        model = AlignedResNet50(False)
        model.train().to(device=self.DEVICE)
        ge = model(x)
        assert ge.shape == (8, 2048)

    def test_aligned_resnet50_train_aligned(self):
        x = torch.randn((8, 3, 224, 224)).to(device=self.DEVICE)
        model = AlignedResNet50(True)
        model.train().to(device=self.DEVICE)
        ge, le = model(x)
        assert ge.shape == (8, 2048)
        assert le.shape == (8, 128, 7)

    def test_aligned_resnet50_eval(self):
        x = torch.randn((8, 3, 224, 224)).to(device=self.DEVICE)
        model = AlignedResNet50()
        model.eval().to(device=self.DEVICE)
        assert model(x).shape == (8, 2048)

    def test_aligned_resnet50_eval_aligned(self):
        x = torch.randn((8, 3, 224, 224)).to(device=self.DEVICE)
        model = AlignedResNet50(True)
        model.eval().to(device=self.DEVICE)
        assert model(x).shape == (8, 2048)
