from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import Market1501, ImageDataset
from src.reranking import ClusterReRanker
from src.models import AlignedResNet50
from tqdm import tqdm


def train(cfg: dict):
    import cv2

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
        for imgs, pids in tqdm(gallery_loader):
            embs = model(imgs.cuda())
            for e, p in zip(embs, pids):
                p = int(p.item())
                if p not in clusters.keys():
                    clusters.update({p: [e]})
                else:
                    clusters[p].append(e)

        for k in clusters.keys():
            clusters[k] = torch.stack(clusters[k], dim=0)

        ranker = ClusterReRanker(clusters=clusters)

        for img, pid in queries:
            pred = model(img.unsqueeze(0).cuda())
            print(pred.shape)

            break


if __name__ == "__main__":
    train(None)
