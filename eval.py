from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import Market1501, ImageDataset
from src.models import AlignedResNet50
from src.engine import evaluate


def eval(cfg: dict):
    path = Path(cfg["dataset"])
    dataset = Market1501(path)

    model = AlignedResNet50()
    ckpt = torch.load(cfg["weights"], weights_only=True)
    model.load_state_dict(ckpt["model"])

    gallery = ImageDataset(dataset=dataset.gallery)
    gallery_loader = DataLoader(
        dataset=gallery,
        batch_size=8,
        shuffle=False,
        num_workers=cfg["workers"],
        persistent_workers=cfg["persistent"],
    )

    query = ImageDataset(dataset=dataset.query)
    query_loader = DataLoader(
        dataset=query,
        batch_size=8,
        shuffle=False,
        num_workers=cfg["workers"],
        persistent_workers=cfg["persistent"],
    )

    loss, violations, topk_acc, cluster_acc = evaluate(
        model=model, query_loader=query_loader, gallery_loader=gallery_loader
    )

    print(
        f"Val:  Loss:{loss}  Total Violations:{violations}  TopK ACC:{topk_acc:.4f}  Cluster ACC:{cluster_acc:.4f}"
    )


if __name__ == "__main__":
    path = Path("cfg/config.yaml")

    with open(path) as fd:
        try:
            cfg = yaml.safe_load(fd)
        except yaml.YAMLError as e:
            print(f"Failed to load the YAML file: {e}")
            exit(1)

    eval(cfg=cfg)
