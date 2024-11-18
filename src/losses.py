import torch
from torch import Tensor, nn


def triplet_semi_hard_negative_mining(
    embeddings: Tensor, pids: Tensor, margin: float = 0.5, reduction: str = "mean"
) -> Tensor:
    """computes triplet loss with semi hard-negative mining"""
    device = embeddings.device
    triplet_count = 0
    batch_size = embeddings.size(0)
    loss = torch.tensor(0.0, device=device)
    # cartesian product
    pairwise_dists = torch.cdist(x1=embeddings, x2=embeddings, p=2)

    for i in range(batch_size):
        #  anchor = embeddings[i]
        # combinatorial
        p_mask = (pids == pids[i]) & (torch.arange(batch_size, device=device) > i)
        n_mask = pids != pids[i]

        p_dists = pairwise_dists[i][p_mask]
        n_dists = pairwise_dists[i][n_mask]

        if len(p_dists) == 0 or len(n_dists) == 0:
            continue

        for p_dist in p_dists:
            # formula: https://arxiv.org/pdf/1503.03832
            _mask = (p_dist < n_dists) & (n_dists < (p_dist + margin))
            semi_hard_negatives = n_dists[_mask]

            n = len(semi_hard_negatives)

            if n > 0:
                triplet_count += n
                y = torch.ones_like(semi_hard_negatives, device=device)
                # assert semi_hard_negative_min.item() > p_dist.item()
                loss += nn.functional.margin_ranking_loss(
                    input1=semi_hard_negatives,
                    input2=p_dist.repeat(n),
                    target=y,
                    margin=margin,
                    reduction="sum",
                )

    if triplet_count == 0:
        # print("no violations")
        loss += pairwise_dists[0][0]
        return loss, triplet_count

    if reduction == "mean":
        return loss / triplet_count, triplet_count
    elif reduction == "sum":
        return loss.sum(), triplet_count
    else:
        raise NotImplementedError(f"Unsupported reduction mode: {reduction}")


def shortest_path(dist_mat: Tensor) -> Tensor:
    """computes the shortest matching local features between two embeddings
    Args:
        dist_mat: distance matrix with shape (batch, n, m)
    Retuns:
        dist: shortest distance value in tensor
    """
    m, n = dist_mat.size()[:2]
    dist = [[0 for _ in range(n)] for _ in range(m)]
    # shortest path with dynamic programming
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:
                dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
    dist = dist[-1][-1]
    return dist


def batched_local_euclidean(x: Tensor, y: Tensor) -> Tensor:
    """computes local distances
    Args:
        x: local features with shape (batch, dim, n)
        y: local features with shape (batch, dim, m)
    returns:
        dists: batched local distances with shape (batch,)
    """
    x = x.permute(0, 2, 1)
    y = y.permute(0, 2, 1)
    dists = torch.cdist(x, y, p=2)
    dists = (torch.exp(dists) - 1.0) / (torch.exp(dists) + 1.0)
    # print(dists)
    dists = shortest_path(dists.permute(1, 2, 0))
    return dists


def aligned_triplet_semi_hard_negative_mining(
    embeddings: Tensor,
    local_embeddings: Tensor,
    pids: Tensor,
    global_margin: float = 0.5,
    local_margin: float = 0.5,
    reduction: str = "mean",
) -> Tensor:
    """computes triplet loss with global and local distances
    Args:
        embeddings: global embeddings with shape (batch, dim)
        local_embeddings: local embeddings with shape (batch, dim, local rank)
        pids: ground truth labels for ids
        global_margin: margin for triplet loss with global features
        local_margin: margin for triplet loss with local features
        reduction: loss reduction methods, mean | sum
    Returns:
        a tuple of loss and margin global margin violation count
    """
    device = embeddings.device
    triplet_count = 0
    loss = torch.tensor(0.0, device=device)
    batch_size, dim, n = local_embeddings.shape
    # global distances
    pairwise_dists = torch.cdist(x1=embeddings, x2=embeddings, p=2)
    # local distances
    temp = []

    for le in local_embeddings:
        le = le.unsqueeze(0).expand(batch_size, dim, n)
        dist = batched_local_euclidean(le, local_embeddings)
        temp.append(dist)

    pairwise_local_dists = torch.stack(temp)

    for i in range(batch_size):
        #  anchor = embeddings[i]
        p_mask = (pids == pids[i]) & (torch.arange(batch_size, device=device) > i)
        n_mask = pids != pids[i]

        p_dists = pairwise_dists[i][p_mask]
        n_dists = pairwise_dists[i][n_mask]

        p_local_dists = pairwise_local_dists[i][p_mask]
        n_local_dists = pairwise_local_dists[i][n_mask]
        assert p_dists.shape == p_local_dists.shape

        if len(p_dists) == 0 or len(n_dists) == 0:
            continue

        for p_dist, p_local_dist in zip(p_dists, p_local_dists):
            # formula: https://arxiv.org/pdf/1503.03832
            _mask = (p_dist < n_dists) & (n_dists < (p_dist + global_margin))
            semi_hard_negatives = n_dists[_mask]
            local_semi_hard_negatives = n_local_dists[_mask]

            n = len(semi_hard_negatives)

            if n > 0:
                triplet_count += n
                assert semi_hard_negatives.shape == local_semi_hard_negatives.shape
                y = torch.ones_like(semi_hard_negatives, device=device)
                # assert semi_hard_negative_min.item() > p_dist.item()
                global_loss = nn.functional.margin_ranking_loss(
                    input1=semi_hard_negatives,
                    input2=p_dist.repeat(n),
                    target=y,
                    margin=global_margin,
                    reduction="sum",
                )

                local_loss = nn.functional.margin_ranking_loss(
                    input1=local_semi_hard_negatives,
                    input2=p_local_dist.repeat(n),
                    target=y,
                    margin=local_margin,
                    reduction="sum",
                )

                loss += global_loss + local_loss

    if triplet_count == 0:
        # print("no violations")
        loss += pairwise_dists[0][0]
        return loss, triplet_count

    if reduction == "mean":
        return loss / triplet_count, triplet_count
    elif reduction == "sum":
        return loss.sum(), triplet_count
    else:
        raise NotImplementedError(f"Unsupported reduction mode: {reduction}")
