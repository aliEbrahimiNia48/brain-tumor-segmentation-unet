"""
Step 2b - PyTorch Dataset for the 2D slices made by prepare_data.py.

Same pattern as learnpytorch.io chapter 04: __init__, __len__, __getitem__.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BrainSliceDataset(Dataset):
    def __init__(self, slices_dir: str, split: str, augment: bool = False):
        self.dir = Path(slices_dir)
        self.augment = augment
        cases = json.loads((self.dir / "splits.json").read_text())[split]

        # mmap_mode="r": the file stays on disk; only the slice we ask for is read.
        # This keeps RAM usage low on a laptop.
        self.images = {c: np.load(self.dir / f"{c}_img.npy", mmap_mode="r") for c in cases}
        self.labels = {c: np.load(self.dir / f"{c}_lbl.npy", mmap_mode="r") for c in cases}

        # one global index -> (case, slice number)
        self.index = [(c, i) for c in cases for i in range(len(self.labels[c]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        case, i = self.index[idx]
        image = torch.from_numpy(np.array(self.images[case][i], dtype=np.float32))  # (4, H, W)
        label = torch.from_numpy(np.array(self.labels[case][i], dtype=np.int64))  # (H, W)

        if self.augment:
            # left-right flip: brains are roughly symmetric, so this is a safe augmentation.
            # The SAME flip must be applied to the image and its label.
            if torch.rand(1) < 0.5:
                image, label = image.flip(-1), label.flip(-1)
            # small random intensity change, so the model does not memorize exact values
            image = image * (1 + 0.1 * (torch.rand(1) - 0.5)) + 0.1 * (torch.rand(1) - 0.5)

        return image, label


if __name__ == "__main__":
    # quick sanity check:  python src/dataset.py
    ds = BrainSliceDataset("data/slices", "train", augment=True)
    x, y = ds[0]
    print("slices in train:", len(ds))
    print("image:", x.shape, x.dtype, "| label:", y.shape, y.dtype, "| classes:", y.unique().tolist())
