import argparse
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from model.Unet_model import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from evaluation import *


PRED_MODEL = 'epoch_19_dsc_0.4594_best_val_dcsc.pth'
IMG_INPUT = 'data/FA/test/imgs/'
GT_INPUT = 'data/FA/test/masks/'


DICE_SAVE = 'output/model_output/'
MASK_SAVE = os.path.join(DICE_SAVE,'predict/')
PROB_SAVE = os.path.join(DICE_SAVE,'probs/')


THRESHOLD = 0.5
SCALE = 0.3


CM_SAVE= DICE_SAVE
# Confusion_Matrix

def predict_img(net,
                full_img,
                true_mask,
                device,
                size,
                scale_factor=1,
                out_threshold=0.5,
                save_dir = PROB_SAVE):
    net.eval()

    b, _, w, h = full_img.shape

    img = full_img
    mask = true_mask

    img = img.to(device=device, dtype=torch.float32)
    mask = mask.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # output,att_map = net(img)
        output = net(img)

        
        probs = torch.sigmoid(output)
        pred = (probs > 0.5).float()
        # probs = probs.squeeze(1)

        save_array = probs.cpu().numpy()
        prob=np.reshape(save_array,[b,w,h])      
        

        probs = F.interpolate(probs, size=size[0],mode='bilinear')
        full_mask = probs.squeeze().cpu().numpy()
        # att_maps = att_map.squeeze().cpu().numpy()

        DSC = [dice_coeff(img_pred, img_true).item() for img_pred, img_true in zip( pred, mask)]
        PC = [get_precision(img_pred, img_true).item() for img_pred, img_true in zip( pred, mask)]
        SE = [get_sensitivity(img_pred, img_true).item() for img_pred, img_true in zip( pred, mask)]
        SP = [get_specificity(img_pred, img_true).item() for img_pred, img_true in zip( pred, mask)]
        F1 = [get_F1(img_pred, img_true).item() for img_pred, img_true in zip( pred, mask)]
       
        TP = [get_TP(img_pred, img_true).item() for img_pred, img_true in zip( pred, mask)]
        FP = [get_FP(img_pred, img_true).item() for img_pred, img_true in zip( pred, mask)]
        TN = [get_TN(img_pred, img_true).item() for img_pred, img_true in zip( pred, mask)]
        FN = [get_FN(img_pred, img_true).item() for img_pred, img_true in zip( pred, mask)]

    return full_mask > out_threshold, prob, DSC, TP, FP, TN, FN, SE, SP, PC


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default= PRED_MODEL,
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                            help='filenames of input images', default = IMG_INPUT)
    parser.add_argument('--gt', '-g', metavar='GT', nargs='+',
                            help='filenames of mask images', default = GT_INPUT)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default= THRESHOLD)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default= SCALE)

    return parser.parse_args()

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))




if __name__ == "__main__":
    if not os.path.isdir(MASK_SAVE):
        os.makedirs(MASK_SAVE)
    args = get_args()
    mask_type = torch.float32

    net = UNet(n_channels=1, n_classes=1, bilinear= False)
    img = BasicDataset(IMG_INPUT, GT_INPUT, SCALE)
    loader = DataLoader(img, batch_size = 5, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    
    AC_array = []
    AC_data= []


    
    for batch_i, batch in enumerate(loader):
        imgs, true_masks = batch['image'], batch['mask']
        img_names, mask_names = batch['image_name'], batch['mask_name']
        true_masks = true_masks.to(device=device, dtype=mask_type)
        n_val = len(loader)
        # for Windows
        #name = [data.split('\\')[1] for data in img_names]
        
        # for ubuntu
        name = [data.split('/')[-1] for data in img_names]
        img_size = batch['original_size']

        masks, prob, DSC, TP, FP, TN, FN, SE, SP, PC = predict_img(net=net,
                                    full_img=imgs,
                                    true_mask=true_masks,
                                    size = img_size,
                                    scale_factor=SCALE,
                                    out_threshold=THRESHOLD,
                                    device=device)
        
        if not args.no_save:
            index = 0
            for mask in masks:
                out_fn = MASK_SAVE + name[index][:-4] + '_OUT' + name[index][-4:]
                result = mask_to_image(mask)
                result.save(out_fn)

                index += 1

    