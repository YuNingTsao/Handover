import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluation import *
from sklearn.metrics import average_precision_score, f1_score,recall_score,precision_score
from sklearn.utils.multiclass import type_of_target
import sys

import os
import numpy as np
from torchvision import transforms
from PIL import Image

def get_average_precision_score(preds, true_masks):
    
    ap = 0
    for i, pred in enumerate(preds):
        
        prob = pred.squeeze(0).cpu().numpy()
        prob = prob.reshape(prob.shape[0] * prob.shape[1],1)
        
        gt = true_masks[i].squeeze(0).cpu().numpy()
        gt = gt.reshape(gt.shape[0] * gt.shape[1],1)
        gt[gt > 0] = 1
          
        ap += average_precision_score(gt,prob)
    
    return ap / ( i + 1)
    
from medpy import metric
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0
def test_single_volume(label, output, classes=2):
    label = torch.clamp(label, 0, 1)
    label = label.squeeze(0).cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            output == i, label == i))
    
    return metric_list

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    
    tot_ac = 0
    tot_pc = 0
    tot_se = 0
    tot_sp = 0
    tot_ap = 0
    tot = 0
    global true_masks, masks_pred 
    
    metric_list = 0.0

    with tqdm(total=n_val, desc='Validation round', unit='img',ascii=True,ncols=120) as pbar:
    # with tqdm(total=n_val, desc='Validation round', unit='img') as pbar:
        for batch in loader:
            imgs, true_masks, ids = batch['image'], batch['mask'], batch['image_name']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                # mask_pred, att_map = net(imgs)
                mask_pred = net(imgs)
                      
            pred = torch.sigmoid(mask_pred)   
            pred = (pred > 0.5).float()

            name = [data.split('/')[-1] for data in ids]
            folder_name = None
            folder_name = os.path.join("see_image")
            os.makedirs(folder_name, exist_ok=True)

            for i in range(0, int(pred.size(0))):
                folder_test_data = os.path.join(folder_name, "test_data")
                os.makedirs(folder_test_data, exist_ok=True)
                image_test_data = transforms.ToPILImage()(imgs[i])
                image_test_data.save(os.path.join(folder_test_data, str(name[i]) + "_test_data.png"))

                folder_test_t = os.path.join(folder_name, "test_t")
                os.makedirs(folder_test_t, exist_ok=True)
                # image_val_t = Image.fromarray(np.uint8(true_masks[i].detach().cpu().numpy()))
                image_val_t = transforms.ToPILImage()(true_masks[i])
                image_val_t.save(os.path.join(folder_test_t, str(name[i]) + "_test_t.png"))

                folder_test_prob = os.path.join(folder_name, "test_prob")
                # image_test_prob = pred[i].squeeze()
                # image_test_prob = torch.argmax(image_test_prob, dim=0).cpu().numpy()
                image_test_prob = pred[i][i].detach().cpu().numpy()
                image_test_prob = Image.fromarray((image_test_prob * 255).astype(np.uint8))
                image_test_prob.save(os.path.join(folder_test_prob, str(name[i]) + "_test_prod.png"))

            tot += dice_coeff(pred, true_masks).item()
            tot_ac += get_accuracy(pred, true_masks)
            tot_pc += get_precision(pred, true_masks)
            tot_se += get_sensitivity(pred, true_masks)
            tot_sp += get_specificity(pred, true_masks)
            tot_ap += get_average_precision_score(pred, true_masks)
            pbar.update(1)
            # print(pred.shape)
            # print(mask_pred.shape)
            # print(true_masks.shape)
            out = pred.squeeze(0)
            out = out.squeeze(0)
            # print(spred.sum())
            # out = spred.squeeze(0)
            # out = out.squeeze(0)
            # out = torch.argmax(torch.softmax(spred, dim=1), dim=1).squeeze(0)
            # print(out.sum())
            # out = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0)
            # print(out.shape)
            # print(true_masks.shape)
            label = true_masks.squeeze(0)
            # print(label.shape)
            metric_i = test_single_volume(label, out, classes=2)
            metric_list += np.array(metric_i)
        metric_list = metric_list / n_val
        # index_mDice = np.mean(metric_list, axis=0)[0]
        index_mean_hd95 = np.mean(metric_list, axis=0)[1]
            
    net.train()

    return tot / n_val, tot_ac/n_val, tot_pc/n_val, tot_se/n_val, tot_sp/n_val, tot_ap/n_val,\
           index_mean_hd95, \
           true_masks, mask_pred
