                      
                       
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
from monai.networks.nets import UNETR
import csv
import os
import numpy as np
import scipy.ndimage as ndimage
from medpy import metric
from tqdm import tqdm
import argparse
import nibabel as nib

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser(description='')
          
parser.add_argument('--root', type=str, default='./data/LITS/crop',
                    help='Directory containing imagesTr/labelsTr and imagesVal/labelsVal')
parser.add_argument('--output_parameter', type=str, default='./parameters/HM_LITS.pth',
                    help='')
parser.add_argument('--dataset', type=str, default='Task03_Liver', help='')
parser.add_argument('--img_size', type=int, nargs='+', default=[128,128,128], help='3D ROI')
parser.add_argument('--n_channels', type=int, default=1, help='')
           
parser.add_argument('--network', type=str, default='HM', help='')
                                                                                     
parser.add_argument('--mode', type=str, default='validation', help='train/validation/test')
parser.add_argument('--ds', default=False, help='')
parser.add_argument('--pretrained_weights', default='', help='')
parser.add_argument('--pretrain_classes', default='', help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--crop_sample', type=int, default=2, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--optim', type=str, default='AdamW', help='')
parser.add_argument('--max_iter', type=int, default=60000, help='')
parser.add_argument('--eval_step', type=int, default=200, help='')
parser.add_argument('--val_batch', type=int, default=2, help='')
parser.add_argument('--overlap', type=float, default=0.5, help='')
parser.add_argument('--overlap_mode', type=str, default='gaussian', help='')
        
parser.add_argument('--gpu', type=str, default='1', help='')
parser.add_argument('--cache_rate', type=float, default=0, help='')
parser.add_argument('--num_workers', type=int, default=1, help='')
      
parser.add_argument('--resume', action='store_true', default=True, help='')
                         
parser.add_argument('--save_dir', type=str, default='./outputs/HM-Net',
                    help='validation')
args = parser.parse_args()
if args.mode == 'validation':
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[INFO] Created output directory: {args.save_dir}")
    else:
        print(f"[INFO] Output directory exists: {args.save_dir}")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

        
train_samples, valid_samples, out_classes = data_loader(args)

set_determinism(seed=0)

train_transforms, val_transforms = data_transforms(args)

            
if args.mode == 'train':
                    
    train_files = [{"image": i, "label": l} for i, l in zip(train_samples["images"], train_samples["labels"])]
    val_files = [{"image": i, "label": l} for i, l in zip(valid_samples["images"], valid_samples["labels"])]

    assert len(train_files) > 0, "train_files is empty. Please check the imagesTr/labelsTr folders."
    assert len(val_files) > 0, "val_files is empty. Please check the imagesVal/labelsVal folders."

    print(f'Training samples: {len(train_files)}')
    print(f'Validation samples: {len(val_files)}')

elif args.mode in ['validation', 'test']:
                    
    train_files = []
    val_files = [{"image": i, "label": l} for i, l in zip(valid_samples["images"], valid_samples["labels"])]

    assert len(val_files) > 0, "val_files is empty. Please check the validation-set path."
    print(f'Validation samples: {len(val_files)}')
else:
    raise ValueError(f"Unsupported mode: {args.mode}")

print('Start caching datasets!')

                 
if args.mode == 'train':
    train_ds = CacheDataset(train_files, train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
else:
    train_ds = None
    train_loader = None

val_ds = CacheDataset(val_files, val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
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

elif args.network == 'HM':
    from networks.Heiix_Mamba.HM import UXNET
    model = UXNET(
        in_chans=args.n_channels,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        spatial_dims=3
    ).to(device)

elif args.network == 'Unetr':

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
elif args.network == 'PHNet':
    from networks.phnet import PHNet

    model = PHNet(
        res_ratio=5 / 0.74,
        layers=(15, 4),
        in_channels=args.n_channels,
        out_channels=out_classes,
        embed_dims=(42, 84, 168, 168, 336),
        segment_dim=(8, 8),
        mlp_ratio=4.0,
        dropout_rate=0.2,
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


            with amp.autocast('cuda', enabled=False):
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

        with amp.autocast('cuda', enabled=False):
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
                    loss += loss_function(
                        F.interpolate(pred, (y.shape[-3], y.shape[-2], y.shape[-1]), mode='trilinear'), y)
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
                print("##################  Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                    dice_val_best, dice_val))
            else:
                print("Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best,
                                                                                                      dice_val))
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
    print(f"Warning: {args.output_parameter} not found, using current model state.")

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


def save_nifti(pred_np, affine, save_path, dtype=None, verbose=False):
    """"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                     
    if dtype is None:
                                              
        if pred_np.max() <= 255 and pred_np.min() >= 0 and np.allclose(pred_np, pred_np.astype(int)):
            dtype = np.uint8
        else:
            dtype = np.int16

            
    if dtype == np.uint8:
        data_to_save = pred_np.astype(np.uint8)
    elif dtype == np.int16:
        data_to_save = pred_np.astype(np.int16)
    elif dtype == np.float32:
        data_to_save = pred_np.astype(np.float32)
    else:
        data_to_save = pred_np.astype(dtype)

    nii_img = nib.Nifti1Image(data_to_save, affine)
    nib.save(nii_img, save_path)

    if verbose:
        print(f"Saved: {save_path} (dtype={dtype}, range=[{data_to_save.min():.2f}, {data_to_save.max():.2f}])")


def validation_emcad(epoch_iterator_val, save_predictions=False):
    """"""
    model.eval()
    dice_vals = []
    hd_vals = []
    per_class_dice = []
    per_class_hd = []
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_iou = []

    case_results = []
    all_file_paths = []

    print(f"\n{'=' * 70}")
    print(f"Starting validation on {len(epoch_iterator_val)} samples")
    print(f"{'=' * 70}\n")

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)

                     
            meta_dict = batch.get("image_meta_dict", {})
            case_name = None
            full_path = None

                           
            if "filename_or_obj" in meta_dict:
                full_path = meta_dict["filename_or_obj"]
                if isinstance(full_path, (list, tuple)):
                    full_path = full_path[0]
                case_name = os.path.basename(full_path)
                case_name = os.path.splitext(os.path.splitext(case_name)[0])[0]
                all_file_paths.append(full_path)

                                  
            if case_name is None and step < len(val_files):
                full_path = val_files[step]["image"]
                case_name = os.path.basename(full_path)
                case_name = os.path.splitext(os.path.splitext(case_name)[0])[0]
                all_file_paths.append(full_path)

                     
            if case_name is None:
                case_name = f"case_{step:04d}"
                print(f"[WARNING] Could not extract original filename for step {step}, using {case_name}")

                          
            if step < 5:
                if full_path:
                    parent_dir = os.path.basename(os.path.dirname(full_path))
                    print(f"[DEBUG] Step {step}: {parent_dir}/{case_name}.nii.gz")
                    print(f"        Full path: {full_path}")
                else:
                    print(f"[DEBUG] Step {step}: {case_name} (no path available)")

                          
            if "affine" in meta_dict:
                affine = meta_dict["affine"]
                if isinstance(affine, (list, tuple)):
                    affine = affine[0]
                affine = affine.cpu().numpy()
            elif "original_affine" in meta_dict:
                affine = meta_dict["original_affine"]
                if isinstance(affine, (list, tuple)):
                    affine = affine[0]
                affine = affine.cpu().numpy()
            else:
                affine = np.eye(4)

            with amp.autocast('cuda', enabled=False):
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

                                                     
                                   
                val_image_normalized = val_inputs.cpu().numpy()[0, 0, :, :, :]

                                
                                                                            
                                 
                                                   
                val_image_display = val_image_normalized.copy()

                                                   
                                                  
                                              
                val_image_display = np.clip(val_image_display * 300, -1000, 1000).astype(np.int16)
                                            

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

                                  
                case_results.append({
                    'case_name': case_name,
                    'dice': mean_dice,
                    'hd': mean_hd,
                    'prediction': val_outputs_np,
                    'affine': affine,
                    'image': val_image_display,             
                    'image_normalized': val_image_normalized,               
                    'label': val_labels_np,
                    'full_path': full_path
                })

                epoch_iterator_val.set_description(
                    "Validate (%d Steps) (mean_dice=%2.5f mean_hd=%2.5f)" %
                    (step, np.mean(dice_vals), np.mean(hd_vals))
                )

    mean_dice_val = np.mean(dice_vals)
    mean_hd_val = np.mean(hd_vals)

                             

            
    print(f"\n{'=' * 70}")
    print(f"Validation Complete - File Source Verification:")
    print(f"{'=' * 70}")

    if all_file_paths:
        val_count = sum(1 for p in all_file_paths if 'imagesVal' in p)
        train_count = sum(1 for p in all_file_paths if 'imagesTr' in p)
        test_count = sum(1 for p in all_file_paths if 'imagesTs' in p)

        print(f"Total files processed: {len(all_file_paths)}")
        print(f"  - From imagesVal: {val_count}")
        print(f"  - From imagesTr: {train_count}")
        print(f"  - From imagesTs: {test_count}")

        print(f"\nSample filenames (first 5):")
        for i, case in enumerate(case_results[:5]):
            print(f"  {i + 1}. {case['case_name']}")
            if case['full_path']:
                print(f"     Path: {case['full_path']}")

        if train_count > 0:
            print(f"\nWARNING: {train_count} training samples found.")
        else:
            print(f"\nAll {val_count} samples are from the validation set.")
    else:
        print("WARNING: Could not extract file paths from metadata.")
        print(f"Total cases processed: {len(case_results)}")
        print(f"\nUsing alternative case names (first 5):")
        for i, case in enumerate(case_results[:5]):
            print(f"  {i + 1}. {case['case_name']}")

    print(f"{'=' * 70}\n")

            
    if save_predictions and args.mode == 'validation':
        save_top_and_worst_cases(case_results)

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


def save_top_and_worst_cases(case_results):
    """"""
    sorted_cases = sorted(case_results, key=lambda x: x['dice'])
    worst_10 = sorted_cases[:10]
    best_10 = sorted_cases[-10:]

    os.makedirs(args.save_dir, exist_ok=True)
    worst_dir = os.path.join(args.save_dir, 'worst_10')
    best_dir = os.path.join(args.save_dir, 'best_10')
    os.makedirs(worst_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Saving validation results to: {args.save_dir}")
    print(f"{'='*70}")

                
    print("\n" + "="*70)
    print("Saving WORST 10 cases (lowest Dice):")
    print("="*70)
    for i, case in enumerate(worst_10):
        prefix = f"val_{case['case_name']}"

                                
        pred_path = os.path.join(worst_dir, f"{prefix}_pred_dice{case['dice']:.4f}.nii.gz")
        save_nifti(case['prediction'], case['affine'], pred_path, dtype=np.uint8)

                               
        label_path = os.path.join(worst_dir, f"{prefix}_label.nii.gz")
        save_nifti(case['label'], case['affine'], label_path, dtype=np.uint8)

                               
        image_path = os.path.join(worst_dir, f"{prefix}_image.nii.gz")
        save_nifti(case['image'], case['affine'], image_path, dtype=np.int16, verbose=True)

        print(f"  [{i+1}/10] {case['case_name']}: Dice={case['dice']:.4f}, HD95={case['hd']:.4f}")

                
    print("\n" + "="*70)
    print("Saving BEST 10 cases (highest Dice):")
    print("="*70)
    for i, case in enumerate(best_10):
        prefix = f"val_{case['case_name']}"

        pred_path = os.path.join(best_dir, f"{prefix}_pred_dice{case['dice']:.4f}.nii.gz")
        save_nifti(case['prediction'], case['affine'], pred_path, dtype=np.uint8)

        label_path = os.path.join(best_dir, f"{prefix}_label.nii.gz")
        save_nifti(case['label'], case['affine'], label_path, dtype=np.uint8)

        image_path = os.path.join(best_dir, f"{prefix}_image.nii.gz")
        save_nifti(case['image'], case['affine'], image_path, dtype=np.int16, verbose=True)

        print(f"  [{i+1}/10] {case['case_name']}: Dice={case['dice']:.4f}, HD95={case['hd']:.4f}")

                  
    csv_path = os.path.join(args.save_dir, 'validation_top_worst_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Rank', 'Case_Name', 'Dice', 'HD95', 'Prediction_File', 'Label_File', 'Image_File'])

        for i, case in enumerate(worst_10):
            prefix = f"val_{case['case_name']}"
            writer.writerow([
                'Worst', i+1, case['case_name'],
                f"{case['dice']:.6f}", f"{case['hd']:.6f}",
                f"worst_10/{prefix}_pred_dice{case['dice']:.4f}.nii.gz",
                f"worst_10/{prefix}_label.nii.gz",
                f"worst_10/{prefix}_image.nii.gz"
            ])

        for i, case in enumerate(best_10):
            prefix = f"val_{case['case_name']}"
            writer.writerow([
                'Best', i+1, case['case_name'],
                f"{case['dice']:.6f}", f"{case['hd']:.6f}",
                f"best_10/{prefix}_pred_dice{case['dice']:.4f}.nii.gz",
                f"best_10/{prefix}_label.nii.gz",
                f"best_10/{prefix}_image.nii.gz"
            ])

    print(f"\nDetailed CSV summary saved to: {csv_path}")
    print("="*70 + "\n")

    print(f"\nStatistics:")
    print(f"  Worst 10 Dice range: {worst_10[0]['dice']:.4f} - {worst_10[-1]['dice']:.4f}")
    print(f"  Best 10 Dice range:  {best_10[0]['dice']:.4f} - {best_10[-1]['dice']:.4f}")
    print(f"  Total cases processed: {len(case_results)}")


                          
save_preds = (args.mode == 'validation')
(
    per_class_dice, per_class_hd, mean_dice_val, mean_hd_val,
    per_class_precision, per_class_recall, per_class_f1, per_class_iou,
    mean_precision, mean_recall, mean_f1, mean_iou
) = validation_emcad(epoch_iterator_val, save_predictions=save_preds)


def print_metrics_to_terminal(trained_weights, dataset_name, network_name, overlap, overlap_mode,
                              class_labels, per_class_dice, per_class_hd, mean_dice_val, mean_hd_val,
                              per_class_precision, per_class_recall, per_class_f1, per_class_iou,
                              mean_precision, mean_recall, mean_f1, mean_iou):
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)

          
    print(f"\nModel:  {os.path.basename(trained_weights)}")
    print(f"Data:   {dataset_name}")
    print(f"Net:    {network_name}")
    print(f"Config: overlap={overlap}, mode={overlap_mode}")

                 
    print("\n" + "-" * 50)
    metrics = [
        ("Dice", mean_dice_val),
        ("HD95", mean_hd_val),
        ("Precision", mean_precision),
        ("Recall", mean_recall),
        ("F1/DSC", mean_f1),
        ("IoU/JI", mean_iou)
    ]

    for metric_name, metric_value in metrics:
        print(f"{metric_name:<20} {metric_value:>10.4f}")

    print("-" * 50 + "\n")


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
