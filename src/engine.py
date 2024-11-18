import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import (
    triplet_semi_hard_negative_mining,
    aligned_triplet_semi_hard_negative_mining,
)
from src.reranking import ReRanker, ClusterReRanker


def train_one_epoch(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    data_loader: DataLoader,
    scaler: torch.GradScaler | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    margin: float = 0.5,
    aligned_loss: bool = False,
    loccal_margin: float = 0.5,
    device: torch.device = torch.device("cuda"),
) -> tuple[float, int]:
    """ train the model for one epoch
    Args:
        model: torch's model
        optim: torch's optimizer
        data_loader: torch's data loader on train dataset
        scaler: GradScaler for automatic mixed-precision
        lr_scheudler: learning rate scheduler
        margin: global margin for triplet loss
        alinged_loss: use local distances if true
        local_margin: margin for local triplet loss
        device: torch's device type
    Returns:
        a tuple of train loss and margin violation count
    """
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
        imgs: Tensor = imgs.to(device=device)
        pids: Tensor = pids.to(device=device)

        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):

            if aligned_loss:
                ge, le = model(imgs)
                loss, count = aligned_triplet_semi_hard_negative_mining(
                    embeddings=ge,
                    local_embeddings=le,
                    pids=pids,
                    global_margin=margin,
                    local_margin=loccal_margin,
                )
            else:
                embs = model(imgs)
                loss, count = triplet_semi_hard_negative_mining(
                    embeddings=embs, pids=pids, margin=margin
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
) -> dict[int, Tensor]:
    """ creates a vector store with id-tensor pairs
    Args:
        model: embedding model
        data_loader: data loader for gallery data
        device: torch's device type
    Retuns:
        vector_store: {pid: embeddings}. embedding shape: (batch, )
    """
    model.eval()
    model.to(device=device)

    vector_store: dict[int, list] = {}

    with torch.no_grad():
        for imgs, pids in tqdm(data_loader):
            imgs: Tensor = imgs.to(device=device)
            embs: Tensor = model(imgs)

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
    topk: int = 5,
    knn: bool = True,
    device: torch.device = torch.device("cuda"),
) -> tuple[float, int, float, float]:
    """evaluates the model performance with re-rankers

    TopK re-ranker -> re-ranking with either top-k or knn
    cluster re-ranker -> re-ranking with average distance

    Args:
        model: torch's model
        query_loader: torch's data loader for query data
        gallery_laoder: torch's data loader for gallery embeddings
        margin: global margin for triplet loss
        topk: k value for top-k
        knn: use knn if true
        device: torch's device type
    Returns:
        a tuple of val loss, margin violation count, top-k accuracy, and cluster accuracy
    """
    model.eval()
    model.to(device=device)

    print("Buliding vector store...")
    vector_store = create_vector_store(
        model=model, data_loader=gallery_loader, device=device
    )

    print("Buliding topk reranker...")
    topk_acc = 0
    topk_reranker = ReRanker(topk=topk)

    for k, x in vector_store.items():
        y = [k for _ in range(x.size(0))]
        topk_reranker.append(x=x, y=y)

    print("Buliding cluster reranker...")
    cluster_acc = 0
    cluster_reranker = ClusterReRanker(clusters=vector_store)

    print("Evaluating rerankers...")
    val_loss = 0.0
    violations = 0

    with torch.no_grad():
        for imgs, pids in tqdm(query_loader):
            imgs: Tensor = imgs.to(device=device)
            pids: Tensor = pids.to(device=device)

            x = model(imgs)
            # local distance is only used for training
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
