from pathlib import Path


class Market1501:
    def __init__(self, root: Path = Path("Market-1501-v15.09.15")):
        self.dataset_dir = root
