from pathlib import Path
from glob import glob
import re


class Market1501:
    def __init__(self, root: Path = Path("Market-1501-v15.09.15")):
        self.dataset_dir = root
        self.train_dir = root / "bounding_box_train"
        self.query_dir = root / "query"
        self.gallery_dir = root / "bounding_box_test"

        self._validate_dir()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _validate_dir(self):
        if not self.dataset_dir.exists():
            raise RuntimeError(f"dataset dir not found: {self.dataset_dir}")
        if not self.train_dir.exists():
            raise RuntimeError(f"train dir not found: {self.train_dir}")
        if not self.query_dir.exists():
            raise RuntimeError(f"query dir not found: {self.query_dir}")
        if not self.gallery_dir.exists():
            raise RuntimeError(f"gallery dir not found: {self.gallery_dir}")

    def _process_dir(self, path: Path, relabel: bool = False) -> tuple[list, int, int]:
        img_paths = glob(str(path / "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d)")

        pid_container = set()
        dataset = []

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            pid_container.add(pid)
            camid -= 1
            dataset.append((img_path, pid, camid))

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            dataset = [
                (img_path, pid2label[pid], camid) for img_path, pid, camid in dataset
            ]

        num_pids = len(pid_container)
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs
