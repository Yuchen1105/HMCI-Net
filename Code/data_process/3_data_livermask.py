import os
import subprocess
import traceback
from tqdm import tqdm


RAW_IMG_DIR = "imagesTr"
TMP_LIVER_DIR = "resampling_liver_mask"

os.makedirs(TMP_LIVER_DIR, exist_ok=True)


def find_liver_mask(case_id):
    case_dir = os.path.join(TMP_LIVER_DIR, case_id)
    if not os.path.exists(case_dir):
        return None
    for root, _, files in os.walk(case_dir):
        for f in files:
            if f.endswith(".nii.gz") and "liver" in f.lower():
                return os.path.join(root, f)
    return None


def run_totalseg(img_path, out_dir, case_id):
    cmd = [
        "TotalSegmentator",
        "-i", img_path,
        "-o", out_dir,
        "--roi_subset", "liver",
        "--device", "gpu",
        "--fast",
    ]
    subprocess.run(cmd)
    return find_liver_mask(case_id)


def main():
    cases = sorted([f for f in os.listdir(RAW_IMG_DIR) if f.endswith(".nii.gz")])

    for case in tqdm(cases):
        try:
            case_id = case.replace(".nii.gz", "")
            liver_out = os.path.join(TMP_LIVER_DIR, case_id)

            liver_mask_path = find_liver_mask(case_id)
            if liver_mask_path is not None:
                continue

            os.makedirs(liver_out, exist_ok=True)
            liver_mask_path = run_totalseg(
                img_path=os.path.join(RAW_IMG_DIR, case),
                out_dir=liver_out,
                case_id=case_id,
            )

            if liver_mask_path is None:
                print("Liver segmentation failed:", case)
                continue

        except Exception:
            print("ERROR:", case)
            traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    main()
