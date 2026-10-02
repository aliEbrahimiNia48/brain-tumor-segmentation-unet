"""
Step 2a - Convert 3D MRI volumes (Task01_BrainTumour) into 2D axial slices.

Why: training a 3D U-Net on a MacBook is too slow. We take 2D slices,
crop the empty border, downsample to 128x128 and save them as .npy files
that can be read lazily (memory-mapped) during training.

Usage:
    python src/prepare_data.py --data-dir data/Task01_BrainTumour --out-dir data/slices --max-cases 200
"""
import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize each modality using only brain (non-zero) voxels.

    volume: (H, W, D, 4)  ->  same shape, float32
    MRI intensities have no fixed scale (unlike CT), so every scan and every
    modality is normalized separately. The black background stays 0.
    """
    out = np.zeros_like(volume, dtype=np.float32)
    for m in range(volume.shape[-1]):
        v = volume[..., m]
        brain = v > 0
        if brain.any():
            mean, std = v[brain].mean(), v[brain].std() + 1e-8
            out[..., m][brain] = (v[brain] - mean) / std
    return out


def select_slices(label: np.ndarray, brain: np.ndarray, n_empty: int, rng: random.Random):
    """Keep every slice that contains tumor + a few brain slices without tumor.

    If we kept only tumor slices, the model would think "there is always a tumor".
    If we kept all slices, most of them would be empty and training would be slow.
    """
    depth = label.shape[2]
    tumor = [z for z in range(depth) if label[:, :, z].any()]
    empty = [z for z in range(depth) if not label[:, :, z].any() and brain[:, :, z].any()]
    empty = rng.sample(empty, min(n_empty, len(empty)))
    return sorted(tumor + empty)


def process_case(img_path: Path, lbl_path: Path, crop: int, size: int, n_empty: int, rng):
    image = nib.load(str(img_path)).get_fdata(dtype=np.float32)  # (240, 240, 155, 4)
    label = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)  # (240, 240, 155)

    image = normalize(image)
    brain = image.any(axis=-1)
    zs = select_slices(label, brain, n_empty, rng)

    # center crop 240 -> crop (the brain sits in the middle, borders are empty)
    h = image.shape[0]
    s = (h - crop) // 2
    image = image[s:s + crop, s:s + crop, zs, :]  # (crop, crop, n, 4)
    label = label[s:s + crop, s:s + crop, zs]  # (crop, crop, n)

    # resize crop -> size. order=1 (linear) for images, order=0 (nearest) for labels,
    # because a label like 2.5 between class 2 and 3 would be meaningless.
    f = size / crop
    image = zoom(image, (f, f, 1, 1), order=1)
    label = zoom(label, (f, f, 1), order=0)

    # to PyTorch layout: (n_slices, channels, H, W) and (n_slices, H, W)
    image = image.transpose(2, 3, 0, 1).astype(np.float16)
    label = label.transpose(2, 0, 1).astype(np.uint8)
    return image, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/Task01_BrainTumour")
    p.add_argument("--out-dir", default="data/slices")
    p.add_argument("--max-cases", type=int, default=200, help="use fewer cases to save time/disk")
    p.add_argument("--crop", type=int, default=192)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--empty-slices", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The tar file contains hidden macOS files like "._BRATS_001.nii.gz" -> skip them
    cases = sorted(p.name.replace(".nii.gz", "")
                   for p in (data_dir / "imagesTr").glob("*.nii.gz")
                   if not p.name.startswith("."))
    rng.shuffle(cases)
    cases = sorted(cases[: args.max_cases])

    # Split by PATIENT, not by slice. Neighbouring slices of one patient look
    # almost identical; mixing them across train/test would leak information.
    shuffled = cases[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    splits = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }

    total = 0
    for case in tqdm(cases, desc="cases"):
        image, label = process_case(
            data_dir / "imagesTr" / f"{case}.nii.gz",
            data_dir / "labelsTr" / f"{case}.nii.gz",
            args.crop, args.size, args.empty_slices, rng,
        )
        np.save(out_dir / f"{case}_img.npy", image)
        np.save(out_dir / f"{case}_lbl.npy", label)
        total += len(label)

    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved {total} slices from {n} cases to {out_dir}")
    print({k: len(v) for k, v in splits.items()}, "cases per split")


if __name__ == "__main__":
    main()
