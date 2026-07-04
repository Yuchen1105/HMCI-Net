import json
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


IMG_DIR = Path("/image")
LBL_DIR = Path("/mask")
OUT_IMG_DIR = Path("/imagesTr")
OUT_LBL_DIR = Path("/labelsTr")

FORCE_TARGET_SPACING = (0.703, 0.703, 1.5)

def read_spacing(img_path):
    img = sitk.ReadImage(str(img_path))
    return img.GetSpacing()                

def compute_spacing_stats(img_dir):
    spacings = []
    files = sorted(img_dir.glob("*.nii.gz"))
    for f in tqdm(files, desc="Collecting spacings", total=len(files)):
        spacings.append(read_spacing(f))
    arr = np.array(spacings, dtype=float)
    df = pd.DataFrame(arr, columns=["sx", "sy", "sz"])
    stats = {
        "n": len(spacings),
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "median": df.median().to_dict(),
        "min": df.min().to_dict(),
        "max": df.max().to_dict(),
    }
    return stats, df

def resample_volume(img, target_spacing, is_label=False):
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    orig_direction = img.GetDirection()
    orig_origin = img.GetOrigin()

    target_spacing = tuple(map(float, target_spacing))
    target_size = [
        int(round(osz * (ospc / tspc)))
        for osz, ospc, tspc in zip(orig_size, orig_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(orig_direction)
    resampler.SetOutputOrigin(orig_origin)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampled = resampler.Execute(img)
    if is_label:
        resampled = sitk.Cast(resampled, sitk.sitkUInt8)
    return resampled

def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)


    stats, df = compute_spacing_stats(IMG_DIR)
    if FORCE_TARGET_SPACING is not None:
        target_spacing = tuple(FORCE_TARGET_SPACING)
    else:
        target_spacing = tuple(stats["median"][k] for k in ["sx", "sy", "sz"])
    print("Spacing stats:", json.dumps(stats, indent=2))
    print(f"Target spacing: {target_spacing}")

    with open(OUT_IMG_DIR / "spacing_stats.json", "w") as f:
        json.dump({"stats": stats, "target_spacing": target_spacing}, f, indent=2)
    df.to_csv(OUT_IMG_DIR / "spacing_all_cases.csv", index=False)

    img_files = sorted(IMG_DIR.glob("*.nii.gz"))
    missing = []
    for img_file in tqdm(img_files, desc="Resampling volumes", total=len(img_files)):
        case = img_file.stem.replace(".nii", "")
        lbl_file = LBL_DIR / f"{case}.nii.gz"
        if not lbl_file.exists():
            missing.append(case)
            continue

        img = sitk.ReadImage(str(img_file))
        lbl = sitk.ReadImage(str(lbl_file))

        img_res = resample_volume(img, target_spacing, is_label=False)
        lbl_res = resample_volume(lbl, target_spacing, is_label=True)

        sitk.WriteImage(img_res, str(OUT_IMG_DIR / f"{case}.nii.gz"))
        sitk.WriteImage(lbl_res, str(OUT_LBL_DIR / f"{case}.nii.gz"))

    if missing:
        print("Missing labels for cases:", missing)
    print("Done.")

if __name__ == "__main__":
    main()