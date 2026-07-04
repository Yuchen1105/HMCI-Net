                      
                       
from torch import amp

from monai.utils import set_determinism                 
from monai.transforms import AsDiscrete, Compose, Activations      
from monai.metrics import DiceMetric, HausdorffDistanceMetric      
from monai.losses import DiceCELoss      
from monai_utils.inferers.utils import sliding_window_inference_1out          
from monai.data import CacheDataset, DataLoader, decollate_batch               
        
from monai.losses import TverskyLoss
import torch
from torch.utils.tensorboard import SummaryWriter       
from load_datasets_transforms import data_loader, data_transforms               
from monai.networks.blocks import UnetOutBlock       
import torch.nn as nn
import torch.nn.functional as F

import csv
import os
import numpy as np
import scipy.ndimage as ndimage
from medpy import metric
from tqdm import tqdm
import argparse

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser(description='Task03_Liver training without dataset.json')

parser.add_argument('--root', type=str, default='./data/LITS/crop', help='Directory containing imagesTr/labelsTr and imagesVal/labelsVal')
parser.add_argument('--output_parameter', type=str, default='HM_LITS.pth', help='')
parser.add_argument('--dataset', type=str, default='Task03_Liver', help='')
parser.add_argument('--img_size', type=int, nargs='+', default=[128,128,128], help='3D ROI')
parser.add_argument('--n_channels', type=int, default=1, help='')

parser.add_argument('--network', type=str, default='HM', help='')

parser.add_argument('--mode', type=str, default='train', help='train/validation/test')
parser.add_argument('--ds', default=False, help='Use deep supervision with multi-scale outputs')
parser.add_argument('--pretrained_weights', default='', help='Path to pretrained or resume checkpoint')
parser.add_argument('--pretrain_classes', default='', help='Number of output classes in the pretrained model')
parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
parser.add_argument('--crop_sample', type=int, default=2, help='Number of cropped sub-volumes per sample')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer type')
parser.add_argument('--max_iter', type=int, default=70000, help='Maximum number of training iterations')
parser.add_argument('--eval_step', type=int, default=200, help='Validation interval in training steps')
parser.add_argument('--val_batch', type=int, default=2, help='Sliding-window batch size during validation')
parser.add_argument('--overlap', type=float, default=0.5, help='Sliding-window overlap ratio')
parser.add_argument('--gpu', type=str, default='0', help='GPU index')
parser.add_argument('--cache_rate', type=float, default=0, help='Dataset cache rate')
parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
parser.add_argument('--resume', action='store_true', default=True, help='Resume training from checkpoint if available')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))


train_samples, valid_samples, out_classes = data_loader(args)

set_determinism(seed=0)

train_transforms, val_transforms = data_transforms(args)


train_files = [{"image": i, "label": l} for i, l in zip(train_samples["images"], train_samples["labels"])]
val_files   = [{"image": i, "label": l} for i, l in zip(valid_samples["images"], valid_samples["labels"])]

assert len(train_files) > 0, "train_files is empty. Please check the imagesTr/labelsTr folders."
assert len(val_files) > 0, "val_files is empty. Please check the imagesVal/labelsVal folders."

print('Start caching datasets!')
train_ds = CacheDataset(train_files, train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
val_ds   = CacheDataset(val_files,   val_transforms,   cache_rate=args.cache_rate, num_workers=args.num_workers)

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=True,
)

if args.ds == 'True':           
    args.ds = True

               
device = torch.device("cuda:0")
if args.network == '3DUXNET':
    from networks.UXNet_3D.network_backbone import UXNET
    model = UXNET(
        in_chans=args.n_channels,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)
elif args.network == 'Unetr':
    from monai.networks.nets import UNETR
    model = UNETR(
        in_channels=args.n_channels,
        out_channels=out_classes,
        img_size=tuple(args.img_size),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        dropout_rate=0.0,
        spatial_dims=3,
        qkv_bias=False,
                                                          
    ).to(device)
elif args.network == 'SegFormer3D':
    from networks.SegFormer3D.segformer3d import SegFormer3D
    model = SegFormer3D(
        in_channels=args.n_channels,
        sr_ratios=[4, 2, 1, 1],
        embed_dims=[32, 64, 160, 256],
        patch_kernel_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        decoder_head_embedding_dim=256,
        num_classes=out_classes,
        decoder_dropout=0.0,
    ).to(device)
elif args.network == 'SegMamba':
    from networks.segmamba import SegMamba
    model = SegMamba(args.n_channels, out_classes).to(device)
elif args.network == 'Zig_RiR3d':
    from networks.Zig_RiR3d import Z_RiR
    model = Z_RiR(in_channels=args.n_channels, out_channels=out_classes).to(device)
elif args.network == 'EMNet':
    from networks.em_net_model import EMNet
    model = EMNet(
        in_chans=args.n_channels,
        out_chans=out_classes,
        depths=[3, 3, 3, 3],
        feat_size=[48, 96, 192, 384],
        hidden_size=384,
        fft_nums=[0, 0, 0, 0],
        conv_decoder=False,
        in_shpae=[128, 128, 128],
    ).to(device)
elif args.network == 'UKAN_EP':
    from networks.ukanep import UKAN
    model = UKAN(input_channnel=args.n_channels, num_classes=out_classes).to(device)
elif args.network == 'UNETVL':
    from networks.UNETVL import UNETR_LSTM
    model = UNETR_LSTM(input_dim=args.n_channels, output_dim=out_classes).to(device)
elif args.network == 'SuperLightNet':
    from networks.superlightnet import NormalU_Net
    model = NormalU_Net(init_channels=args.n_channels, class_nums=out_classes, depths_unidirectional='small').to(device)
elif args.network == 'WNet':
    from networks.nnWNet import WNet3D
    model = WNet3D(in_channel=args.n_channels, num_classes=out_classes, deep_supervised=False).to(device)
elif args.network == 'SwinSMT':
    from networks.SwinSMT import SwinSMT
    model = SwinSMT(
        in_channels=args.n_channels,
        out_channels=out_classes,
        img_size=(128, 128, 128),
        spatial_dims=3,
        use_v2=False,
        feature_size=48,
        use_moe=True,
        num_experts=4,
        num_layers_with_moe=2,
    ).to(device)
elif args.network == 'HM':
    from networks.Heiix_Mamba.MsMambaV04 import UXNET
    model = UXNET(
        in_chans=args.n_channels,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        spatial_dims=3
    ).to(device)
else:
    raise ValueError(f"Unsupported network: {args.network}")

print('Chosen Network Architecture: {}'.format(args.network))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)


loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: DiceCELoss')


if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    raise ValueError(f"Unsupported optimizer: {args.optim}")
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))

scaler = amp.GradScaler('cuda')

global_step = 0
dice_val_best = 0.0
global_step_best = 0


def load_checkpoint_if_needed():
    global global_step, dice_val_best, global_step_best
    if args.resume and os.path.isfile(args.pretrained_weights):
        ckpt = torch.load(args.pretrained_weights, map_location=device)
        state_dict = ckpt.get("model", ckpt)                              
                            
        new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("global_step", global_step)
        dice_val_best = ckpt.get("dice_val_best", dice_val_best)
        global_step_best = ckpt.get("global_step_best", global_step_best)
        print(f"[Resume] Loaded checkpoint from {args.pretrained_weights} "
              f"(global_step={global_step}, best_dice={dice_val_best:.4f})")
    elif args.resume:
        print(f"[Resume] Checkpoint {args.pretrained_weights} not found, start fresh.")

load_checkpoint_if_needed()

def forward_for_infer(x):
    p = model(x)
    if isinstance(p, torch.Tensor):
        return p
    elif isinstance(p, (list, tuple)):
        return p[0]
    elif isinstance(p, dict):
        return p["out"]
    else:
        raise TypeError(f"Unexpected model output type: {type(p)}")

def validation(epoch_iterator_val, max_batches=50):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            if step >= max_batches:
                break
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

            with amp.autocast('cuda', enabled=True):
                val_outputs = sliding_window_inference_1out(
                    val_inputs,
                    (args.img_size[0], args.img_size[1], args.img_size[2]),
                    args.val_batch,
                    forward_for_infer,
                    overlap=args.overlap,
                )

                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)

                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f, mean_dice=%2.5f)" %
                    (global_step, 10.0, dice, np.mean(dice_vals))
                )
            dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    return mean_dice_val

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())


        with amp.autocast('cuda', enabled=True):
            p = model(x)
            if isinstance(p, torch.Tensor):
                P = [p]
            elif isinstance(p, (list, tuple)):
                P = list(p)
            else:
                raise TypeError(f"Unexpected model output type: {type(p)}")

            loss = 0.0
            if args.ds is True:
                for pred in P:
                    loss += loss_function(F.interpolate(pred, (y.shape[-3], y.shape[-2], y.shape[-1]), mode='trilinear'), y)
            else:
                loss = loss_function(F.interpolate(P[0], (y.shape[-3], y.shape[-2], y.shape[-1]), mode='trilinear'), y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        optimizer.zero_grad()

        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "global_step": global_step,
                        "dice_val_best": dice_val_best,
                        "global_step_best": global_step_best,
                        "args": vars(args),
                    },
                    args.output_parameter,
                )
                print("##################  Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
            else:
                print("Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
        global_step += 1
    return global_step, dice_val_best, global_step_best

max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))

eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
epoch_loss_values = []
metric_values = []

    
if args.mode == 'train':
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)

    
if os.path.isfile(args.output_parameter):
    ckpt = torch.load(os.path.join(args.output_parameter), map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in state_dict.items()}, strict=False)
    print(f"Loaded best checkpoint from {args.output_parameter} for evaluation.")
else:
    model.load_state_dict(torch.load(os.path.join(args.output_parameter)))
model.eval()

post_label = AsDiscrete(to_onehot=out_classes)
post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=out_classes)])
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
hd_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False)

epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def calculate_prf_iou(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_dice = 2 * precision * recall / (precision + recall + 1e-7)
    iou_ji = tp / (tp + fp + fn + 1e-7)
    return precision, recall, f1_dice, iou_ji

def validation_emcad(epoch_iterator_val):
    model.eval()
    dice_vals = []
    hd_vals = []
    per_class_dice = []
    per_class_hd = []
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_iou = []

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)

            with amp.autocast('cuda', enabled=True):
                val_outputs = sliding_window_inference_1out(
                    val_inputs,
                    (args.img_size[0], args.img_size[1], args.img_size[2]),
                    args.val_batch,
                    forward_for_infer,
                    overlap=args.overlap,
                    mode=args.overlap_mode,
                )

                val_labels_np = val_labels.cpu().numpy()[0, 0, :, :, :]
                val_outputs_np = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs_np = np.argmax(val_outputs_np, axis=1).astype(np.uint8)[0]
                val_outputs_np = resample_3d(val_outputs_np, target_shape)

                dice_list_sub = []
                hd_list_sub = []
                prec_list_sub = []
                rec_list_sub = []
                f1_list_sub = []
                iou_list_sub = []
                for i in range(1, out_classes):
                    cls_pred = (val_outputs_np == i)
                    cls_gt = (val_labels_np == i)
                    cls_dice, cls_hd = calculate_metric_percase(cls_pred, cls_gt)
                    dice_list_sub.append(cls_dice)
                    hd_list_sub.append(cls_hd)

                    p, r, f1, iou = calculate_prf_iou(cls_pred, cls_gt)
                    prec_list_sub.append(p)
                    rec_list_sub.append(r)
                    f1_list_sub.append(f1)
                    iou_list_sub.append(iou)

                mean_dice = np.mean(dice_list_sub)
                mean_hd = np.mean(hd_list_sub)

                per_class_dice.append(dice_list_sub)
                per_class_hd.append(hd_list_sub)
                per_class_precision.append(prec_list_sub)
                per_class_recall.append(rec_list_sub)
                per_class_f1.append(f1_list_sub)
                per_class_iou.append(iou_list_sub)

                dice_vals.append(mean_dice)
                hd_vals.append(mean_hd)

                epoch_iterator_val.set_description(
                    "Validate (%d Steps) (mean_dice=%2.5f mean_hd=%2.5f mean_f1=%2.5f mean_iou=%2.5f)" %
                    (step, np.mean(dice_vals), np.mean(hd_vals), np.mean(f1_list_sub), np.mean(iou_list_sub))
                )

    mean_dice_val = np.mean(dice_vals)
    mean_hd_val = np.mean(hd_vals)
    return (
        np.array(per_class_dice),
        np.array(per_class_hd),
        mean_dice_val,
        mean_hd_val,
        np.array(per_class_precision),
        np.array(per_class_recall),
        np.array(per_class_f1),
        np.array(per_class_iou),
        np.array(per_class_precision).mean(),
        np.array(per_class_recall).mean(),
        np.array(per_class_f1).mean(),
        np.array(per_class_iou).mean(),
    )

(
    per_class_dice, per_class_hd, mean_dice_val, mean_hd_val,
    per_class_precision, per_class_recall, per_class_f1, per_class_iou,
    mean_precision, mean_recall, mean_f1, mean_iou
) = validation_emcad(epoch_iterator_val)

def print_metrics_to_terminal(trained_weights, dataset_name, network_name, overlap, overlap_mode,
                              class_labels, per_class_dice, per_class_hd, mean_dice_val, mean_hd_val,
                              per_class_precision, per_class_recall, per_class_f1, per_class_iou,
                              mean_precision, mean_recall, mean_f1, mean_iou):
    print("\n==================== Validation Metrics ====================")
    print(f"Trained Weights: {trained_weights}")
    print(f"Dataset       : {dataset_name}")
    print(f"Network       : {network_name}")
    print(f"Overlap       : {overlap}")
    print(f"Overlap Mode  : {overlap_mode}")
    print("-------------------------------------------------------------")

    class_names = [class_labels[str(i)] for i in range(len(class_labels))]
    class_names_no_first = class_names[1:]
    header = " | ".join([f"{name:^10}" for name in class_names_no_first])

    print("\nDice per Class:")
    print(header)
    print("-" * len(header))
    avg_dice_per_class = per_class_dice.mean(axis=0)
    print(" | ".join([f"{dice:.4f}".center(10) for dice in avg_dice_per_class]))
    print(f"Mean Dice: {mean_dice_val:.4f}")

    print("\nHD per Class:")
    print(header)
    print("-" * len(header))
    avg_hd_per_class = per_class_hd.mean(axis=0)
    print(" | ".join([f"{hd:.4f}".center(10) for hd in avg_hd_per_class]))
    print(f"Mean HD: {mean_hd_val:.4f}")

    print("\nPrecision per Class:")
    print(header)
    print("-" * len(header))
    avg_prec_per_class = per_class_precision.mean(axis=0)
    print(" | ".join([f"{p:.4f}".center(10) for p in avg_prec_per_class]))
    print(f"Mean Precision: {mean_precision:.4f}")

    print("\nRecall per Class:")
    print(header)
    print("-" * len(header))
    avg_rec_per_class = per_class_recall.mean(axis=0)
    print(" | ".join([f"{r:.4f}".center(10) for r in avg_rec_per_class]))
    print(f"Mean Recall: {mean_recall:.4f}")

    print("\nF1/DSC per Class:")
    print(header)
    print("-" * len(header))
    avg_f1_per_class = per_class_f1.mean(axis=0)
    print(" | ".join([f"{f1:.4f}".center(10) for f1 in avg_f1_per_class]))
    print(f"Mean F1/DSC: {mean_f1:.4f}")

    print("\nIoU/JI per Class:")
    print(header)
    print("-" * len(header))
    avg_iou_per_class = per_class_iou.mean(axis=0)
    print(" | ".join([f"{iou:.4f}".center(10) for iou in avg_iou_per_class]))
    print(f"Mean IoU/JI: {mean_iou:.4f}")
    print("=============================================================\n")

                               
class_labels = {
    "0": "background",
    "1": "Tumor",
}

print_metrics_to_terminal(
    args.output_parameter,
    args.dataset,
    args.network,
    args.overlap,
    args.overlap_mode,
    class_labels,
    per_class_dice,
    per_class_hd,
    mean_dice_val,
    mean_hd_val,
    per_class_precision,
    per_class_recall,
    per_class_f1,
    per_class_iou,
    mean_precision,
    mean_recall,
    mean_f1,
    mean_iou,
)
