from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import Market1501, ImageDataset
from src.models import AlignedResNet50
from src.engine import train_one_epoch


def train(cfg: dict):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if cfg["market1501"]:
        dataset = Market1501(Path(cfg["dataset"]))
    else:
        raise NotImplementedError()

    train = ImageDataset(dataset=dataset.train)
    train_loader = DataLoader(
        dataset=train,
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=cfg["workers"],
        persistent_workers=cfg["persistent"],
    )

    model = AlignedResNet50(aligned=cfg["aligned"])

    match cfg["optimizer"].lower():
        case "sgd":
            sgd_cfg = cfg["sgd"]
            optim = torch.optim.SGD(
                params=model.parameters(),
                lr=cfg["lr"],
                momentum=sgd_cfg["momentum"],
                weight_decay=sgd_cfg["decay"],
                nesterov=sgd_cfg["nesterov"],
            )
        case "adamw":
            adamw_cfg = cfg["adamw"]
            optim = torch.optim.AdamW(
                params=model.parameters(),
                lr=cfg["lr"],
                betas=adamw_cfg["betas"],
                weight_decay=adamw_cfg["decay"],
            )
        case _:
            raise NotImplementedError(f"Unsupported optimizer: {cfg['optimizer']}")

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim, step_size=cfg["step"], gamma=cfg["gamma"]
    )

    scaler = torch.GradScaler() if cfg["scaler"] else None

    epochs = cfg["epochs"]

    for e in range(epochs):
        print(f"----- Epoch {e} ----- at LR: {lr_scheduler.get_last_lr()[-1]}")

        loss, count = train_one_epoch(
            model=model,
            optim=optim,
            data_loader=train_loader,
            epoch=e,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            margin=cfg["margin"],
            aligned_loss=cfg["aligned"],
            device=device,
        )

        print(f"Train:  Loss:{loss}  Margin violations:{count}")

    print("Done!")


if __name__ == "__main__":
    path = Path("cfg/config.yaml")

    with open(path) as fd:
        try:
            cfg = yaml.safe_load(fd)
        except yaml.YAMLError as e:
            print(f"Failed to load the YAML file: {e}")
            exit(1)

    train(cfg=cfg)
