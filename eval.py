from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import Market1501, ImageDataset
from src.engine import evaluate
from src.models import AlignedResNet50


def eval(cfg: dict):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    path = Path(cfg["dataset"])
    dataset = Market1501(path)

    model = AlignedResNet50(cfg["aligned"])
    ckpt = torch.load(cfg["weights"], weights_only=True)
    model.load_state_dict(ckpt["model"])

    gallery = ImageDataset(dataset=dataset.gallery)
    gallery_loader = DataLoader(
        dataset=gallery,
        batch_size=128,
        shuffle=False,
        num_workers=cfg["workers"],
        persistent_workers=False,
    )

    query = ImageDataset(dataset=dataset.query)
    query_loader = DataLoader(
        dataset=query,
        batch_size=128,
        shuffle=False,
        num_workers=cfg["workers"],
        persistent_workers=False,
    )

    loss, violations, topk_acc, cluster_acc = evaluate(
        model=model,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        margin=cfg["global_margin"],
        topk=cfg["topk"],
        knn=cfg["knn"],
        device=device,
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
