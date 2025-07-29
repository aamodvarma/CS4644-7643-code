# main_finetune_structured.py
import os
import json
import time
import datetime
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
#assert timm.__version__ == "0.3.2"

import models_mae
from datasets.masked_faces_dataset import MaskedFacesDataset
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune_structured import train_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('MAE finetuning with structured masks', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1.5e-4)
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--output_dir', default='./output_dir', type=str)
    parser.add_argument('--log_dir', default='./output_dir', type=str)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--finetune', default='./mae_pretrain_vit_base.pth', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    print("Job dir:", os.path.dirname(os.path.realpath(__file__)))
    print(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    image_dir = os.path.join(args.data_path, 'train', 'images')
    mask_dir = os.path.join(args.data_path, 'train', 'masks_structured_variant')
    mask_variants = ['masks_structured-e', 'masks_structured-nm', 'masks_structured-all', 'masks_edge_50']

    filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    dataset_train = MaskedFacesDataset(image_dir, mask_dir, filenames, mask_variants)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    if args.finetune and os.path.isfile(args.finetune):
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Loaded pretrained model from", args.finetune)

    eff_batch_size = args.batch_size * args.accum_iter
    args.lr = args.lr or args.blr * eff_batch_size / 256

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    log_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args
        )

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            misc.save_model(
                args=args, model=model, model_without_ddp=model,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps({**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}) + "\n")

    total_time = time.time() - start_time
    print("Training time:", str(datetime.timedelta(seconds=int(total_time))))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
