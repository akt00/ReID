import torch
from torch import Tensor, nn


def triplet_semi_hard_negative_mining(
    embeddings: Tensor, pids: Tensor, margin: float = 0.5, reduction: str = "mean"
) -> Tensor:
    device = embeddings.device

    loss = torch.tensor(0.0, device=device)
    triplet_count = 0

    batch_size = embeddings.size(0)
    pairwise_dists = torch.cdist(x1=embeddings, x2=embeddings, p=2)

    for i in range(batch_size):
        #  anchor = embeddings[i]
        p_mask = (pids == pids[i]) & (torch.arange(batch_size, device=device) > i)
        n_mask = (pids != pids[i])

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
        loss += pairwise_dists[0][0]
        return loss, triplet_count

    if reduction == "mean":
        return loss / triplet_count, triplet_count
    elif reduction == "sum":
        return loss.sum(), triplet_count
    else:
        raise NotImplementedError(f"Unsupported reduction mode: {reduction}")
