"""
train.py — Training script for:
  "Boundary-Guided Camouflaged Object Detection"
  IEEE Transactions on Circuits and Systems for Video Technology, 2024
  DOI: 10.1109/TCSVT.2024.10483928

Usage:
    python train.py --train_path /path/to/TrainDataset

Expected dataset structure:
    TrainDataset/
        Imgs/   — RGB images
        GT/     — binary ground-truth masks
        Edge/   — binary edge maps

Checkpoints are saved to:  checkpoints/<train_save>/
Training log is written to: log/BGNet.txt
"""

import os
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Network
from net.network import Network

# Data & utilities
from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 4444
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

# ── GPU selection ──────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"   # adjust as needed


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def structure_loss(pred, mask):
    """
    Weighted IoU + weighted BCE loss.
    Boundary regions receive higher weights (×5) to encourage sharp predictions.

    Reference: F3Net (Wei et al., AAAI 2020).
    """
    weight  = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce    = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce    = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred    = torch.sigmoid(pred)
    inter   = ((pred * mask) * weight).sum(dim=(2, 3))
    union   = ((pred + mask) * weight).sum(dim=(2, 3))
    wiou    = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def dice_loss(pred, target):
    """
    Soft Dice loss — used for the auxiliary EAM mask output.
    Expects *probability* predictions (already sigmoid-activated).
    """
    smooth = 1.0
    p      = 2
    pred   = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    num    = (pred * target).sum(dim=1) * 2 + smooth
    den    = (pred.pow(p) + target.pow(p)).sum(dim=1) + smooth
    return (1 - num / den).mean()


# ---------------------------------------------------------------------------
# Training loop (single epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(train_loader, model, optimizer, epoch, opt, log_file):
    model.train()

    meters = {
        'mask': AvgMeter(),
        'o3':   AvgMeter(),
        'o2':   AvgMeter(),
        'o1':   AvgMeter(),
    }

    for step, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        # ── Prepare data ─────────────────────────────────────────────────
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts    = Variable(gts).cuda()
        edges  = Variable(edges).cuda()

        # ── Forward pass ─────────────────────────────────────────────────
        o3, o2, o1, mask = model(images)

        # ── Compute losses ────────────────────────────────────────────────
        loss_o3   = structure_loss(o3,   gts)
        loss_o2   = structure_loss(o2,   gts)
        loss_o1   = structure_loss(o1,   gts)
        loss_mask = dice_loss(mask, gts)          # mask is already sigmoid-activated

        loss = loss_o3 + loss_o2 + loss_o1 + loss_mask

        # ── Backward pass ────────────────────────────────────────────────
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        # ── Update meters ────────────────────────────────────────────────
        bs = opt.batchsize
        meters['o3'].update(loss_o3.item(),   bs)
        meters['o2'].update(loss_o2.item(),   bs)
        meters['o1'].update(loss_o1.item(),   bs)
        meters['mask'].update(loss_mask.item(), bs)

        # ── Logging ──────────────────────────────────────────────────────
        if step % 60 == 0 or step == len(train_loader):
            msg = (
                f"{datetime.now()}  "
                f"Epoch [{epoch:03d}/{opt.epoch:03d}]  "
                f"Step [{step:04d}/{len(train_loader):04d}]  "
                f"o3: {meters['o3'].avg:.4f}  "
                f"o2: {meters['o2'].avg:.4f}  "
                f"o1: {meters['o1'].avg:.4f}  "
                f"mask: {meters['mask'].avg:.4f}"
            )
            print(msg)
            log_file.write(msg + '\n')

    # ── Save checkpoint ──────────────────────────────────────────────────
    save_dir = f'checkpoints/{opt.train_save}/'
    os.makedirs(save_dir, exist_ok=True)

    if (epoch + 1) % 30 == 0 or (epoch + 1) == opt.epoch:
        ckpt_path = os.path.join(save_dir, f'BGNet-{epoch}.pth')
        torch.save(model.state_dict(), ckpt_path)
        msg = f'[Checkpoint saved] {ckpt_path}'
        print(msg)
        log_file.write(msg + '\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train BGNet for Camouflaged Object Detection'
    )
    parser.add_argument('--epoch',      type=int,   default=90,
                        help='total training epochs')
    parser.add_argument('--lr',         type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--batchsize',  type=int,   default=24,
                        help='training batch size')
    parser.add_argument('--trainsize',  type=int,   default=416,
                        help='input image resolution')
    parser.add_argument('--clip',       type=float, default=0.5,
                        help='gradient clipping threshold')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset',
                        help='root path to training dataset')
    parser.add_argument('--train_save', type=str,   default='BGNet',
                        help='sub-folder name for saved checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()

    # ── Build model ──────────────────────────────────────────────────────
    model = Network().cuda()
    model = nn.DataParallel(model)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[Model] Total parameters: {n_params:.2f} M')

    # ── Optimiser ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # ── Data loader ──────────────────────────────────────────────────────
    image_root = os.path.join(opt.train_path, 'Imgs')
    gt_root    = os.path.join(opt.train_path, 'GT')
    edge_root  = os.path.join(opt.train_path, 'Edge')

    train_loader = get_loader(
        image_root, gt_root, edge_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize
    )
    print(f'[Data]  {len(train_loader)} steps per epoch')

    # ── Training ─────────────────────────────────────────────────────────
    os.makedirs('log', exist_ok=True)
    log_file = open('log/BGNet.txt', 'a')

    print('[Training] Start')
    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train_one_epoch(train_loader, model, optimizer, epoch, opt, log_file)

    log_file.close()
    print('[Training] Complete')
