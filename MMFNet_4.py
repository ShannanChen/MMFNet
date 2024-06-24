#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:06:19 2021

@author: leeh43
"""
from monai.utils import set_determinism
from monai.transforms import AsDiscrete, Activations, Compose
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, AttentionUnet, SwinUNETR
# from networks.SwinUNETR.swin_unetr import SwinUNETR
from networks.UNet.unet2_F2Net_nolocal_4 import UNet
# from networks.SegResNet.segresnet import SegResNetguiwe
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch

import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms_4 import data_loader, data_transforms

import os
import numpy as np
from tqdm import tqdm
import argparse
import random
from torch import nn
# import networks.factorizer as ft

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='3D UX-Net hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='', required=True, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='', required=True,
                    help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=True,
                    help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='3DUXNET',
                    help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', default='', help='Path of pretrained weights')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=500, help='Per steps to perform validation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

train_samples, valid_samples, out_classes = data_loader(args)

train_files = [
    {"image": image_name, "T1": images_T1, "T2": images_T2, "images_flair": images_flair, "label": label_name}
    for image_name, images_T1, images_T2, images_flair, label_name in
    zip(train_samples['images'], train_samples['T1'], train_samples['T2'], train_samples['images_flair'], train_samples['labels'])
]

val_files = [
    {"image": image_name, "T1": images_T1, "T2": images_T2, "images_flair": images_flair, "label": label_name}
    for image_name, images_T1, images_T2, images_flair, label_name in
    zip(valid_samples['images'], valid_samples['T1'], valid_samples['T2'], valid_samples['images_flair'], valid_samples['labels'])
]

set_determinism(seed=0)

train_transforms, val_transforms = data_transforms(args)

## Train Pytorch Data Loader and Caching
print('Start caching datasets!')
train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=args.cache_rate, num_workers=args.num_workers)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=True)

## Valid Pytorch Data Loader and Caching
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)

val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

## Load Networks
device = torch.device("cuda:0")
if args.network == '3DUXNET':
    model = UXNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)
elif args.network == 'SwinUNETR':
    model = SwinUNETR(
        img_size=(64, 64, 64),
        in_channels=1,
        out_channels=out_classes,
        feature_size=96,
        use_checkpoint=False,
    ).to(device)
elif args.network == 'nnFormer':
    model = nnFormer(input_channels=1, num_classes=out_classes).to(device)

elif args.network == 'UNETR':
    model = UNETR(
        in_channels=1,
        out_channels=out_classes,
        img_size=(64, 64, 64),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)
elif args.network == 'TransBTS':
    _, model = TransBTS(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)
elif args.network == 'UNet':
    model = UNet(n_channels=1, n_classes=out_classes, width_multiplier=1, trilinear=True, use_ds_conv=False).to(device)
elif args.network == 'AttentionUnet':
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)


print('Chosen Network Architecture: {}'.format(args.network))
if args.pretrain == 'True':
    print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights))
    model.load_state_dict(torch.load(args.pretrained_weights))

## Define Loss function and optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
# loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)


root_dir = os.path.join(args.output)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)

t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)


def validation(epoch_iterator_val):
    # model_feat.eval()
    model.eval()
    dice_vals = list()
    Hausdorff_Distance_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_inputs_T1, val_inputs_T2, val_inputs_flair, val_labels = (
                batch["image"].cuda(), batch["T1"].cuda(), batch["T2"].cuda(), batch["images_flair"].cuda(), batch["label"].cuda())
            # val_outputs = model(val_inputs)
            # zero_tensor_x_dwi = torch.zeros_like(val_inputs)
            # zero_tensor_x_T1 = torch.zeros_like(val_inputs_T1)
            # zero_tensor_x_T2 = torch.zeros_like(val_inputs_T2)
            # zero_tensor_x_flair = torch.zeros_like(val_inputs_flair)
            val_inputs = torch.concat((val_inputs, val_inputs_T1, val_inputs_T2, val_inputs_flair), 1)
            val_outputs_fusion, val_outputs_dwi, val_outputs_T1, val_outputs_T2 , val_inputs_flair = sliding_window_inference(val_inputs, (64, 64, 64), 2, model)
            # val_outputs = model_seg(val_inputs, val_feat[0], val_feat[1])
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs_fusion)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            HausdorffDistance(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            Hausdorff_Distance = HausdorffDistance.aggregate().item()
            dice_vals.append(dice)
            Hausdorff_Distance_vals.append(Hausdorff_Distance)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
        HausdorffDistance.reset()
    mean_dice_val = np.mean(dice_vals)
    mean_Hausdorff_Distance_val = np.mean(Hausdorff_Distance_vals)
    writer.add_scalar('Validation Segmentation Loss', mean_dice_val, global_step)
    return mean_dice_val, mean_Hausdorff_Distance_val


def train(global_step, train_loader, dice_val_best, hd_val_best, global_step_best):
    # model_feat.eval()
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, x_T1, x_T2, x_flair, y = (
            batch["image"].cuda(), batch["T1"].cuda(), batch["T2"].cuda(), batch["images_flair"].cuda(), batch["label"].cuda())
        # with torch.no_grad():
        #     g_feat, dense_feat = model_feat(x)
        # zero_tensor_x = torch.zeros_like(x)
        # zero_tensor_x_T1 = torch.zeros_like(x_T1)
        # zero_tensor_x_T2 = torch.zeros_like(x_T2)
        # zero_tensor_x_flair = torch.zeros_like(x_flair)
        x = torch.concat((x, x_T1, x_T2, x_flair), 1)
        logit_map_fusion, logit_map_dwi, logit_map_adc, logit_map_flair, logit_map_3 = model(x)
        logit_map_fusion, logit_map_dwi, logit_map_adc, logit_map_flair, logit_map_3 = model(x)
        loss1 = loss_function(logit_map_fusion, y)
        loss2 = loss_function(logit_map_dwi, y)
        loss3 = loss_function(logit_map_adc, y)
        loss4 = loss_function(logit_map_flair, y)
        loss5 = loss_function(logit_map_3, y)
        loss = loss1 + loss2 + loss3 + loss4 + loss5
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
                global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val, Hausdorff_Distance_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            HD_values.append(Hausdorff_Distance_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                hd_val_best = Hausdorff_Distance_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Current Best Avg. HD: {} Current Avg. HD: {}".format(
                        dice_val_best, dice_val, hd_val_best, Hausdorff_Distance_val
                    )
                )
                # scheduler.step(dice_val)
            else:
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Current Best Avg. HD: {} Current Avg. HD: {}".format(
                        dice_val_best, dice_val, hd_val_best, Hausdorff_Distance_val
                    )
                )
                # scheduler.step(dice_val)
        writer.add_scalar('Training Segmentation Loss', loss.data, global_step)
        global_step += 1
    return global_step, dice_val_best, hd_val_best, global_step_best


max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
HausdorffDistance = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95, get_not_nans=False)
global_step = 0
dice_val_best = 0.0
hd_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
HD_values = []
while global_step < max_iterations:
    global_step, dice_val_best, hd_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, hd_val_best, global_step_best
    )





