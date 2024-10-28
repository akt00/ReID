from torch import nn, Tensor
from torchvision import models


class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x, kernel_size=(1, inp_size[-1]))


class AlignedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.global_extractor = nn.AdaptiveAvgPool2d(1)
        self.local_extractor = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            HorizontalMaxPool2d(),
            nn.Conv2d(
                in_channels=2048,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        emb = self.backbone(x)
        ge = self.global_extractor(emb)
        ge = ge.squeeze(-1)

        if self.training:
            le = self.local_extractor(emb)
            le = le.squeeze(-1)
            return ge, le

        return ge
