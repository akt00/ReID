import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import triplet_semi_hard_negative_mining
from src.reranking import ReRanker, ClusterReRanker


def train_one_epoch(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    data_loader: DataLoader,
    scaler: torch.GradScaler | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    margin: int = 0.5,
    aligned_loss: bool = False,
    device: torch.device = torch.device("cuda"),
) -> tuple[float, int]:

    model.train()
    model.to(device=device)
    """
    warmup_lr = None
    # warmup lr
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        warmup_lr = torch.optim.lr_scheduler.LinearLR(
            optimizer=optim, start_factor=warmup_factor, total_iters=warmup_iters
        )
    """
    violations = 0
    train_loss = 0.0

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

        train_loss += loss.item()
        violations += count
        optim.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer=optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        """
        if warmup_lr is not None:
            warmup_lr.step()
        """
    if lr_scheduler is not None:
        lr_scheduler.step()

    return train_loss / len(data_loader), violations


def create_vector_store(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cuda"),
) -> dict[int, torch.Tensor]:

    model.eval()
    model.to(device=device)

    vector_store: dict[int, list] = {}

    with torch.no_grad():
        for imgs, pids in tqdm(data_loader):
            imgs: torch.Tensor = imgs.to(device=device)
            embs: torch.Tensor = model(imgs)

            for e, p in zip(embs, pids):
                p = int(p.item())

                if p not in vector_store.keys():
                    vector_store.update({p: [e]})
                else:
                    vector_store[p].append(e)

    for k in vector_store.keys():
        vector_store[k] = torch.stack(vector_store[k], dim=0)

    return vector_store


def evaluate(
    model: torch.nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    margin: float = 0.5,
    topk: int = 3,
    knn: bool = True,
    device: torch.device = torch.device("cuda"),
) -> tuple[float, int, float, float]:

    model.eval()
    model.to(device=device)

    print("Buliding vector store...")
    vector_store = create_vector_store(
        model=model, data_loader=gallery_loader, device=device
    )

    topk_acc = 0
    topk_reranker = ReRanker(topk=topk)

    print("Buliding topk reranker...")

    for k, x in vector_store.items():
        y = [k for _ in range(x.size(0))]
        topk_reranker.append(x=x, y=y)

    print("Buliding cluster reranker...")

    cluster_acc = 0
    cluster_reranker = ClusterReRanker(vector_store)

    val_loss = 0.0
    violations = 0

    with torch.no_grad():
        for imgs, pids in tqdm(query_loader):
            imgs: torch.Tensor = imgs.to(device=device)
            pids: torch.Tensor = pids.to(device=device)

            x = model(imgs)

            loss, count = triplet_semi_hard_negative_mining(
                embeddings=x, pids=pids, margin=margin
            )
            val_loss += loss.item()
            violations += count

            res = topk_reranker.evaluate(x, pids, knn=knn)
            topk_acc += res.sum().item()

            res = cluster_reranker.evalute(x, pids)
            cluster_acc += res.sum().item()

    return (
        val_loss / len(query_loader),
        violations,
        topk_acc / len(query_loader.dataset),
        cluster_acc / len(query_loader.dataset),
    )
