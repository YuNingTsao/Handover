import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import copy
from PIL import Image
import os
import cv2

path='./Cross_contraining'
out_path='./Cross_contraining'
# Unet AG-PCVNet Trans-AG-PCVNet TransUnet Deeplabv3+
# MT PS-MT Cross_contraining Deep_cotraining
# Dual-MT Siamese-MT
# target
dirs=os.listdir(path)
dirs.sort()

for file in dirs:
    img = cv2.imread(path+'/'+file,cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img)
    # img=np.flipud(img)
    a1 = copy.deepcopy(img)
    a2 = copy.deepcopy(img)
    a3 = copy.deepcopy(img)
    #terminal=R
    a1[a1 == 255] = 255

    #terminal=G
    a2[a2 == 255] = 0

    #terminal=B
    a3[a3 == 255] = 0


    a1 = Image.fromarray(np.uint8(a1)).convert('L')
    a2 = Image.fromarray(np.uint8(a2)).convert('L')
    a3 = Image.fromarray(np.uint8(a3)).convert('L')
    out_img = Image.merge('RGB', [a1, a2, a3])
    out_img.save(out_path+'/RGB_'+file)
    # img = cv2.cvtColor(np.asarray(out_img),cv2.COLOR_RGB2BGR) 
    # cv2.imshow('RGB',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()