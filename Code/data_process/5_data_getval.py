import os
import shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from collections import defaultdict


IMG_DIR = Path("/imagesTr")
LAB_DIR = Path("/labelsTr")

OUT_IMG_VAL = Path("/imagesVal")
OUT_LAB_VAL = Path("/labelsVal")

OUT_IMG_VAL.mkdir(parents=True, exist_ok=True)
OUT_LAB_VAL.mkdir(parents=True, exist_ok=True)


SEED = 42
VAL_RATIO = 0.2
MOVE_FILES = True

Z_PCTL_MIN = 1
Z_PCTL_MAX = 99


Z_BIN_COUNT = 5

TUMOR_BINS = [0, 1e-4, 1e-3, 1e-2, 1.0]
TUMOR_WEIGHTS = [0.5, 1.0, 2.0, 2.0]

rng = np.random.default_rng(SEED)

def read_shape_and_ratio(img_path, lab_path):
    img = sitk.ReadImage(str(img_path))
    lab = sitk.ReadImage(str(lab_path))
    img_np = sitk.GetArrayFromImage(img)
    lab_np = sitk.GetArrayFromImage(lab)
    z, y, x = img_np.shape
    vox = img_np.size
    nonzero = np.count_nonzero(lab_np)
    ratio = nonzero / vox if vox > 0 else 0.0
    return (z, y, x), ratio

def quantile_bins(values, bin_count):
    qs = np.linspace(0, 1, bin_count + 1)
    edges = np.unique(np.quantile(values, qs))
    if len(edges) <= 2:
        edges = np.array([values.min() - 1e-6, values.max() + 1e-6])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def assign_bin(val, edges):
    return np.searchsorted(edges, val, side="right") - 1

def weighted_stratified_split(meta):
    n_total = len(meta)
    n_val = max(1, int(round(n_total * VAL_RATIO)))


    z_bins = [m["z_bin"] for m in meta]
    unique_bins, counts = np.unique(z_bins, return_counts=True)
    props = counts / counts.sum()
    quotas = np.floor(props * n_val).astype(int)

    short = n_val - quotas.sum()
    for i in range(short):
        quotas[i % len(quotas)] += 1
    quota_map = dict(zip(unique_bins, quotas))

    val_indices = []
    for b in unique_bins:
        idx_in_bin = [i for i, m in enumerate(meta) if m["z_bin"] == b]
        if len(idx_in_bin) == 0:
            continue
        quota = quota_map.get(b, 0)
        if quota <= 0:
            continue

        weights = []
        for i in idx_in_bin:
            tb = meta[i]["tumor_bin"]
            w = TUMOR_WEIGHTS[min(tb, len(TUMOR_WEIGHTS)-1)]
            weights.append(w)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        chosen = rng.choice(idx_in_bin, size=min(quota, len(idx_in_bin)), replace=False, p=weights)
        val_indices.extend(chosen.tolist())


    val_indices = list(dict.fromkeys(val_indices))
    if len(val_indices) < n_val:
        remaining = [i for i in range(n_total) if i not in val_indices]
        need = n_val - len(val_indices)
        extra = rng.choice(remaining, size=need, replace=False)
        val_indices.extend(extra.tolist())

    val_set = set(val_indices)
    train_indices = [i for i in range(n_total) if i not in val_set]
    return train_indices, list(val_set)

def move_or_copy(src, dst):
    if MOVE_FILES:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)

def main():

    imgs = {p.name: p for p in IMG_DIR.glob("*.nii.gz")}
    labs = {p.name: p for p in LAB_DIR.glob("*.nii.gz")}
    common_names = sorted(set(imgs.keys()) & set(labs.keys()))
    if not common_names:
        print("No paired files found.")
        return


    meta = []
    zs = []
    for name in common_names:
        shape, ratio = read_shape_and_ratio(imgs[name], labs[name])
        z, y, x = shape
        meta.append({
            "name": name,
            "z": z,
            "ratio": ratio,
        })
        zs.append(z)


    z_lo = np.percentile(zs, Z_PCTL_MIN)
    z_hi = np.percentile(zs, Z_PCTL_MAX)
    meta = [m for m in meta if z_lo <= m["z"] <= z_hi]
    if not meta:
        print("All samples filtered out; adjust Z_PCTL_MIN/MAX.")
        return


    z_edges = quantile_bins([m["z"] for m in meta], Z_BIN_COUNT)
    for m in meta:
        m["z_bin"] = assign_bin(m["z"], z_edges)
        m["tumor_bin"] = assign_bin(m["ratio"], TUMOR_BINS)


    train_idx, val_idx = weighted_stratified_split(meta)
    train_names = [meta[i]["name"] for i in train_idx]
    val_names   = [meta[i]["name"] for i in val_idx]

    print(f"Total usable pairs: {len(meta)}, Train: {len(train_names)}, Val: {len(val_names)}")
    print(f"Z bins edges: {z_edges}")
    print(f"Tumor bins: {TUMOR_BINS}  weights: {TUMOR_WEIGHTS}")


    for name in val_names:
        move_or_copy(imgs[name], OUT_IMG_VAL / name)
        move_or_copy(labs[name], OUT_LAB_VAL / name)


    with open("train_list.txt", "w") as f:
        f.write("\n".join(train_names))
    with open("val_list.txt", "w") as f:
        f.write("\n".join(val_names))

    print("Done. Validation files placed in:")
    print("  ", OUT_IMG_VAL)
    print("  ", OUT_LAB_VAL)

if __name__ == "__main__":
    main()