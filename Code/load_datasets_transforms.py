import glob
import os

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)


def data_loader(args):
    """Load a LiTS-style dataset from imagesTr/labelsTr and imagesVal/labelsVal."""
    root_dir = args.root
    dataset = args.dataset

    if dataset != "Task03_Liver":
        raise ValueError("This anonymous release includes the LiTS example only. Use --dataset Task03_Liver.")

    out_classes = 3
    print(f"Loading LiTS-style data from: {root_dir}")

    if args.mode == "train":
        train_samples = {
            "images": sorted(glob.glob(os.path.join(root_dir, "imagesTr", "*.nii.gz"))),
            "labels": sorted(glob.glob(os.path.join(root_dir, "labelsTr", "*.nii.gz"))),
        }
        valid_samples = {
            "images": sorted(glob.glob(os.path.join(root_dir, "imagesVal", "*.nii.gz"))),
            "labels": sorted(glob.glob(os.path.join(root_dir, "labelsVal", "*.nii.gz"))),
        }
        print(f"Training samples: {len(train_samples['images'])}")
        print(f"Validation samples: {len(valid_samples['images'])}")
        return train_samples, valid_samples, out_classes

    if args.mode == "validation":
        train_samples = {"images": [], "labels": []}
        valid_samples = {
            "images": sorted(glob.glob(os.path.join(root_dir, "imagesVal", "*.nii.gz"))),
            "labels": sorted(glob.glob(os.path.join(root_dir, "labelsVal", "*.nii.gz"))),
        }
        print(f"Validation samples: {len(valid_samples['images'])}")
        return train_samples, valid_samples, out_classes

    if args.mode == "test":
        test_samples = {
            "images": sorted(glob.glob(os.path.join(root_dir, "imagesTs", "*.nii.gz"))),
        }
        print(f"Test samples: {len(test_samples['images'])}")
        return test_samples, out_classes

    raise ValueError(f"Unsupported mode: {args.mode}. Choose train, validation, or test.")


def data_transforms(args):
    """MONAI transforms for the LiTS example."""
    if args.dataset != "Task03_Liver":
        raise ValueError("This anonymous release includes transforms for --dataset Task03_Liver only.")

    crop_samples = args.crop_sample if args.mode in {"train", "validation"} else None

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=args.img_size,
                pos=1,
                neg=1,
                num_samples=crop_samples,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=args.img_size, mode="constant"),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.2,
                spatial_size=args.img_size,
                rotate_range=(0.0, 0.0, 0.15),
                scale_range=(0.1, 0.1, 0.1),
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensityRanged(keys=["image"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )

    if args.mode in {"train", "validation"}:
        return train_transforms, val_transforms
    return test_transforms
