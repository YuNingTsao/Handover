import argparse
import copy
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
from model import UNet
from evaluation import *
from utils.dataset import BasicDataset
from utils.custom_sampler import CustomSampler
from torch.utils.data import DataLoader

PRED_MODEL = './epoch_32_dsc_0.4996_best_val_dcsc.pth'

test_img = 'data/FA/test/img/'
test_mask = 'data/FA/test/mask/'
# test_img = 'data/FA/val/img/'
# test_mask = 'data/FA/val/mask/'

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=80,
                        help='Number of sochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type = float, nargs='?', default=1e-2,
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default= 0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-g', '--gradient-accumulations', dest='gradient_accumulations', type=int, default= 4,
                        help='gradient accumulations')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)

    val = BasicDataset(test_img, test_mask, args.scale)
    val_loader = DataLoader(val, batch_size = 1, shuffle=False, pin_memory=True, drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    net.to(device=device)
    net.load_state_dict(torch.load(PRED_MODEL, map_location=device))

    val_score, ac, pc, se, sp, ap, hd95, true_masks, masks_pred = eval_net(net, val_loader, device)
    print(f"dsc:{val_score*100}", f"HD:{hd95}")