import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_DIR = Path("data")
LBL_DIR = Path("")
OUT_DIR = Path("./stats_out")


TUMOR_LABEL_VALUE = 1

def get_stats(img_path, lbl_path):
    img_ni = nib.load(img_path)
    lbl_ni = nib.load(lbl_path)
    spacing = np.array(img_ni.header.get_zooms()[:3], dtype=float)           
    shape = np.array(img_ni.shape[:3], dtype=int)
    voxel_vol = float(np.prod(spacing))

    lbl = lbl_ni.get_fdata()
    tumor_mask = (lbl == TUMOR_LABEL_VALUE)

    tumor_vox = int(np.count_nonzero(tumor_mask))
    tumor_ml = tumor_vox * voxel_vol / 1000.0

    return {
        "spacing_x": spacing[0], "spacing_y": spacing[1], "spacing_z": spacing[2],
        "shape_x": shape[0], "shape_y": shape[1], "shape_z": shape[2],
        "slices": shape[2],
        "voxel_vol_mm3": voxel_vol,
        "tumor_vox": tumor_vox,
        "tumor_ml": tumor_ml,
    }

def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing = []
    img_files = sorted(IMG_DIR.glob("*.nii.gz"))
    for img_file in tqdm(img_files, desc="Processing cases"):
        case = img_file.stem.replace(".nii", "")
        lbl_file = LBL_DIR / f"{case}.nii.gz"
        if not lbl_file.exists():
            missing.append(case)
            continue
        stats = get_stats(img_file, lbl_file)
        stats["case"] = case
        records.append(stats)

    df = pd.DataFrame(records)
    df = df[[
        "case",
        "spacing_x","spacing_y","spacing_z",
        "shape_x","shape_y","shape_z","slices",
        "voxel_vol_mm3",
        "tumor_vox","tumor_ml"
    ]]
    csv_path = out_dir / "case_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-case stats: {csv_path}")

    summary = {
        "n_cases": len(df),
        "spacing_mean": df[["spacing_x","spacing_y","spacing_z"]].mean().to_dict(),
        "spacing_std": df[["spacing_x","spacing_y","spacing_z"]].std().to_dict(),
        "slices_mean": df["slices"].mean(),
        "slices_std": df["slices"].std(),
        "tumor_ml_mean": df["tumor_ml"].mean(),
        "tumor_ml_std": df["tumor_ml"].std(),
    }
    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {json_path}")
    if missing:
        print(f"Missing labels for cases: {missing}")

           
    plt.figure(figsize=(10, 6))
    plt.subplot(2,2,1); plt.hist(df["spacing_z"], bins=20); plt.title("Spacing Z (mm)")
    plt.subplot(2,2,2); plt.hist(df["slices"], bins=20); plt.title("Slices per volume")
    plt.subplot(2,2,3); plt.hist(df["tumor_ml"], bins=20); plt.title("Tumor volume (ml)")
    plt.tight_layout()
    fig_path = out_dir / "histograms.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved histograms: {fig_path}")

if __name__ == "__main__":
    main()