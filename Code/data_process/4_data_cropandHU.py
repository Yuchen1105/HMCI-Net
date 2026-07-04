import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


RAW_IMG_DIR = "/imagesTr"
RAW_LAB_DIR = "/labelsTr"
MASK_DIR    = "/resampling_liver_mask"

OUT_IMG_DIR = "/imagesTr"
OUT_LAB_DIR = "/labelsTr"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LAB_DIR, exist_ok=True)


WINDOW_WIDTH  = 350.0     
WINDOW_LEVEL  = 40.0      


def read_nii(path):
    return sitk.ReadImage(path)

def sitk_to_np(img):
    return sitk.GetArrayFromImage(img)             

def np_to_sitk(arr, ref_img):
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref_img)
    return out

def get_bbox(mask_np):
    coords = np.where(mask_np > 0)
    if coords[0].size == 0:
        return None
    zmin, zmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    xmin, xmax = coords[2].min(), coords[2].max()
    return zmin, zmax, ymin, ymax, xmin, xmax

def clamp_bbox_to_img(bbox, shape_zyx):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    Z, Y, X = shape_zyx
    zmin = max(0, zmin); ymin = max(0, ymin); xmin = max(0, xmin)
    zmax = min(Z - 1, zmax); ymax = min(Y - 1, ymax); xmax = min(X - 1, xmax)
    if zmin > zmax or ymin > ymax or xmin > xmax:
        return None
    return zmin, zmax, ymin, ymax, xmin, xmax

def crop_sitk(img, bbox_zyx):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox_zyx
    size  = [int(xmax - xmin + 1), int(ymax - ymin + 1), int(zmax - zmin + 1)]
    index = [int(xmin),           int(ymin),           int(zmin)]
    return sitk.RegionOfInterest(img, size=size, index=index)

def find_mask_path(case_id):
    case_dir = os.path.join(MASK_DIR, case_id)
    if not os.path.exists(case_dir):
        return None
    for root, _, files in os.walk(case_dir):
        for f in files:
            if f.endswith(".nii.gz") and "liver" in f.lower():
                return os.path.join(root, f)
    return None

def is_valid_mask(mask_path, min_voxels=10):
    try:
        arr = sitk_to_np(read_nii(mask_path))
        return np.count_nonzero(arr) >= min_voxels
    except Exception:
        return False

def filter_tumor_with_liver(tumor_np, liver_np):
    return np.where(liver_np > 0, tumor_np, 0)


def phase1_collect_global_bbox_and_crop():
    global_bboxes = []
    cases = sorted([f for f in os.listdir(RAW_IMG_DIR) if f.endswith(".nii.gz")])
    for case in tqdm(cases, desc="Phase1: crop with own mask"):
        case_id = case[:-7]
        mask_path = find_mask_path(case_id)
        if mask_path is None:
            continue
        if not is_valid_mask(mask_path):
            print(f"Invalid/empty liver mask, skip: {case}")
            continue

        img_path = os.path.join(RAW_IMG_DIR, case)
        lab_path = os.path.join(RAW_LAB_DIR, case)
        if not os.path.exists(lab_path):
            print(f"Missing tumor label for {case}")
            continue

        img = read_nii(img_path)
        lab = read_nii(lab_path)
        liver_mask = read_nii(mask_path)

        liver_np = sitk_to_np(liver_mask)
        tumor_np = sitk_to_np(lab)


        tumor_np_filtered = filter_tumor_with_liver(tumor_np, liver_np)
        if np.count_nonzero(tumor_np_filtered) == 0:
            print(f"Tumor has no intrahepatic voxels, skip: {case}")
            continue
        lab = np_to_sitk(tumor_np_filtered.astype(tumor_np.dtype), lab)

        bbox = get_bbox(liver_np)
        if bbox is None:
            print(f"Empty liver mask for {case}")
            continue

        bbox = clamp_bbox_to_img(bbox, liver_np.shape)
        if bbox is None:
            print(f"BBox invalid after clamp for {case}")
            continue

        global_bboxes.append(bbox)

        cropped_img = crop_sitk(img, bbox)
        cropped_lab = crop_sitk(lab, bbox)

        out_img = os.path.join(OUT_IMG_DIR, case)
        out_lab = os.path.join(OUT_LAB_DIR, case)
        sitk.WriteImage(cropped_img, out_img)
        sitk.WriteImage(cropped_lab, out_lab)

    return global_bboxes

def merge_global_bbox(global_bboxes):
    if not global_bboxes:
        return None
    zmins, zmaxs, ymins, ymaxs, xmins, xmaxs = zip(*global_bboxes)
    return (min(zmins), max(zmaxs), min(ymins), max(ymaxs), min(xmins), max(xmaxs))


def phase2_crop_missing_masks(global_bbox):
    if global_bbox is None:
        print("No global bbox available; nothing to crop in phase2.")
        return

    cases = sorted([f for f in os.listdir(RAW_IMG_DIR) if f.endswith(".nii.gz")])
    for case in tqdm(cases, desc="Phase2: crop with global bbox"):
        if os.path.exists(os.path.join(OUT_IMG_DIR, case)):
            continue
        img_path = os.path.join(RAW_IMG_DIR, case)
        lab_path = os.path.join(RAW_LAB_DIR, case)
        if not os.path.exists(lab_path):
            print(f"Missing tumor label for {case}")
            continue

        img = read_nii(img_path)
        lab = read_nii(lab_path)

        shape_zyx = sitk_to_np(img).shape
        bbox = clamp_bbox_to_img(global_bbox, shape_zyx)
        if bbox is None:
            print(f"Global bbox invalid for {case} (after clamp)")
            continue

        cropped_img = crop_sitk(img, bbox)
        cropped_lab = crop_sitk(lab, bbox)

        out_img = os.path.join(OUT_IMG_DIR, case)
        out_lab = os.path.join(OUT_LAB_DIR, case)
        sitk.WriteImage(cropped_img, out_img)
        sitk.WriteImage(cropped_lab, out_lab)


def apply_window_to_outputs():
    w = WINDOW_WIDTH
    l = WINDOW_LEVEL
    w_min = l - w / 2.0
    w_max = l + w / 2.0
    img_paths = sorted([p for p in os.listdir(OUT_IMG_DIR) if p.endswith(".nii.gz")])
    for name in tqdm(img_paths, desc="Apply window/level to cropped images"):
        p = os.path.join(OUT_IMG_DIR, name)
        img = read_nii(p)
        arr = sitk_to_np(img).astype(np.float32)
        arr = np.clip(arr, w_min, w_max)
        arr = arr.astype(sitk_to_np(img).dtype)
        sitk.WriteImage(np_to_sitk(arr, img), p)


def main():
    global_bboxes = phase1_collect_global_bbox_and_crop()
    global_bbox = merge_global_bbox(global_bboxes)
    if global_bbox:
        print("Global bbox (zmin,zmax,ymin,ymax,xmin,xmax):", global_bbox)
    else:
        print("No masks found; cannot compute global bbox.")
    phase2_crop_missing_masks(global_bbox)


    apply_window_to_outputs()

    print("All cases processed.")

if __name__ == "__main__":
    main()
