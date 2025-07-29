import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
from engine_finetune_rand import train_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning (reconstruction w/ random masking)', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Ratio of pixels to mask')

    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--color_jitter', type=float, default=None)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--cutmix', type=float, default=0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')

    parser.add_argument('--finetune', default='', help='Pretrained MAE checkpoint')
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--nb_classes', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser

def main(args):
    print("Running with args:\n", args)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=args.num_workers,
                                                    pin_memory=args.pin_mem, drop_last=True)

    model = models_mae.__dict__[args.model](norm_pix_loss=False)
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Loaded pretrained checkpoint from", args.finetune)
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    log_writer = SummaryWriter(log_dir=args.log_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args
        )

        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=model,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

            # Write epoch stats to log.txt
            log_stats = {f"train_{k}": v for k, v in train_stats.items()}
            log_stats["epoch"] = epoch
            if misc.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
