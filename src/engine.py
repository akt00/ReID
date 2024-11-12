from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import Market1501, ImageDataset
from src.reranking import ClusterReRanker
from src.models import AlignedResNet50
from tqdm import tqdm


def train(cfg: dict):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = Market1501()

    clusters: dict[int, list] = {}

    model = AlignedResNet50()
    model.eval().cuda()

    # cls_ranker = ClusterReRanker(clusters=clusters)
    gallery = ImageDataset(dataset=dataset.gallery)
    gallery_loader = DataLoader(dataset=gallery, batch_size=128, num_workers=2)
    queries = ImageDataset(dataset=dataset.query)

    with torch.no_grad():
        embs_list = []
        pid_list = []
        for imgs, pids in tqdm(gallery_loader):
            embs = model(imgs.cuda())
            embs_list.append(embs)
            pid_list.append(pids)

        embs = torch.concat(embs_list, dim=0)
        pids = torch.concat(pid_list, dim=0)

        for e, p in zip(embs, pids):
            print(e.shape, p.shape)
            p = int(p.item())
            if p not in clusters.keys():
                clusters.update({p: [e]})
            else:
                clusters[p].append(e)

        for k in clusters.keys():
            clusters[k] = torch.stack(clusters[k], dim=0)
        
        for k, v in clusters.items():
            print(k, v.shape)

        ranker = ClusterReRanker(clusters=clusters)

        for img, pid in queries:
            pred = model(img.unsqueeze(0).cuda())
            print(pred.shape)

            break


if __name__ == "__main__":
    train(None)
