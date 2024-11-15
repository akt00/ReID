import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import triplet_semi_hard_negative_mining


def train_one_epoch(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    data_loader: DataLoader,
    epoch: int,
    scaler: torch.GradScaler | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    margin: int = 0.5,
    aligned_loss: bool = False,
    device: torch.device = torch.device("cuda"),
) -> tuple[float, int]:

    model.train()
    model.to(device=device)

    warmup_lr = None
    # warmup lr
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        warmup_lr = torch.optim.lr_scheduler.LinearLR(
            optimizer=optim, start_factor=warmup_factor, total_iters=warmup_iters
        )

    total_violations = 0
    loss = 0.0

    for imgs, pids in tqdm(data_loader):
        imgs: torch.Tensor = imgs.to(device=device)
        pids: torch.Tensor = pids.to(device=device)

        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            preds = model(imgs)

            if aligned_loss:
                raise NotImplementedError()
            else:
                loss, count = triplet_semi_hard_negative_mining(
                    embeddings=preds, pids=pids, margin=margin
                )

        loss += loss.item()
        total_violations += count
        optim.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer=optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        if warmup_lr is not None:
            warmup_lr.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    return loss / len(data_loader), total_violations
