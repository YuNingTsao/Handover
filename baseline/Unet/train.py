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

import random
random.seed(42)

train_img = 'data/FA/train/img/'
train_mask = 'data/FA/train/mask/'
val_img = 'data/FA/val/img/'
val_mask = 'data/FA/val/mask/'

dir_checkpoint = 'checkpoints/'
best_dsc = 0.0
best_epoch = 0
    
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

def train_net(net,
              device,
              best_model_param,
              epochs=80,
              batch_size=8,
              lr=1e-2,
              save_cp=True,
              img_scale=0.5,
              grad_accumulations = 2):

    # Get dataloader
    train = BasicDataset(train_img, train_mask, img_scale)
    val = BasicDataset(val_img, val_mask, img_scale)
    n_train = len(train)
    train_sampler = CustomSampler(train)
    
    
    train_loader = DataLoader(train, batch_size = batch_size, shuffle=False, sampler= train_sampler, pin_memory=True)
    val_loader = DataLoader(val, batch_size = 1, shuffle=False, pin_memory=True, drop_last=True)

    global true_masks, masks_pred , best_dsc, best_epoch

    
    # Optimizer setting
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()  
    min_loss = float('inf')
    
    # Train
    for epoch in range(1,epochs + 1):
        net.train()

        epoch_loss = 0
        epoch_dice_loss = 0
        epoch_bce_loss = 0
        epoch_dsc = 0
        step = 1

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img',ascii=True,ncols=120) as pbar:
            
            for batch_i, batch in enumerate(train_loader):
                 # imgs       : input image(eye image)
                 # true_masks : ground truth
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                # print(imgs.shape)
                # print(true_masks.shape)
                # input()
                masks_pred = net(imgs)
                dsc = dice_coeff(torch.sigmoid(masks_pred), true_masks)
                dice_loss = 1 - dsc
                bce_loss = bce(masks_pred, true_masks)
                
                # calculate loss
                # loss =  0.5*bce_loss + 0.5*dice_loss
                loss =  bce_loss
                epoch_loss += loss.item()
                epoch_bce_loss += bce_loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_dsc  += dsc.item()
                
                # Back propogation
                loss.backward()
                
                if (batch_i + 1) % grad_accumulations == 0 or\
                    (len(train_loader) - batch_i < grad_accumulations and\
                      len(train_loader) - batch_i == 1):
                    optimizer.step()
                    optimizer.zero_grad()
                    
                pbar.update(imgs.shape[0])
                step += 1
        
        val_score, ac, pc, se, sp, ap, hd95, true_masks, masks_pred = eval_net(net, val_loader, device)
        if val_score > best_dsc or (val_score == best_dsc and epoch_loss < min_loss):
            best_dsc = val_score
            best_epoch = epoch
            min_loss = epoch_loss
            best_model_params = copy.deepcopy(net.state_dict())

    torch.save(best_model_params, f'epoch_{best_epoch}_dsc_{best_dsc:.4f}_best_val_dcsc.pth')
    print("Best model name : " + f'epoch_{best_epoch}_dsc_{best_dsc:.4f}_best_val_dcsc.pth')
    

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    # total = sum([param.nelement() for param in net.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))
    # input()
    best_model_params = copy.deepcopy(net.state_dict())

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  best_model_param = best_model_params,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  grad_accumulations = args.gradient_accumulations
                  )
    except KeyboardInterrupt:
        print('Saved interrupt')
        torch.save(best_model_params, f'epoch_{best_epoch}_dsc_{best_dsc:.4f}_best_val_dcsc.pth')
        print("Best model name : " + f'epoch_{best_epoch}_dsc_{best_dsc:.4f}_best_val_dcsc.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
