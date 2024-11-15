from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import Market1501, ImageDataset
from src.models import AlignedResNet50
from src.engine import evaluate


def eval(cfg: dict):
    dataset = Market1501()

    model = AlignedResNet50()
    ckpt = torch.load("best.pth", weights_only=True)
    model.load_state_dict(ckpt["model"])

    gallery = ImageDataset(dataset=dataset.gallery)
    gallery_loader = DataLoader(dataset=gallery, batch_size=128, num_workers=2)

    query = ImageDataset(dataset=dataset.query)
    query_loader = DataLoader(dataset=query, batch_size=128, num_workers=2)

    evaluate(model, query_loader, gallery_loader)


if __name__ == "__main__":
    path = Path("cfg/config.yaml")

    with open(path) as fd:
        try:
            cfg = yaml.safe_load(fd)
        except yaml.YAMLError as e:
            print(f"Failed to load the YAML file: {e}")
            exit(1)

    eval(cfg=cfg)
