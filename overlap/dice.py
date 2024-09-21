import numpy as np
import os
import cv2
from sklearn import metrics, neighbors
from sklearn.metrics import confusion_matrix

def information_index(outputs, targets):
    eps = np.finfo(np.float64).eps
    output = outputs.flatten()
    target = targets.flatten()
    TN, FP, FN, TP = confusion_matrix(target,output).ravel()

    index_MIou =  ( TP / (TP + FP + FN + eps) + TN / (TN + FN + FP + eps) ) / 2
    mean_iou = np.mean(index_MIou)
    index_dice = 2*TP / (2*TP + FP + FN + eps)
    mean_dice = np.mean(index_dice)

    return mean_iou, mean_dice

def count_index(pre, tar):
        path_pre = pre
        path_target = tar

        target = cv2.imread(path_target, cv2.IMREAD_GRAYSCALE)
        _, tar = cv2.threshold(target, 128, 255, cv2.THRESH_BINARY)

        predict = cv2.imread(path_pre, cv2.IMREAD_GRAYSCALE)
        predict = cv2.resize(predict, (224, 224), interpolation=cv2.INTER_AREA)
        _, pre = cv2.threshold(predict, 128, 255, cv2.THRESH_BINARY)

        tIOU, tdice = information_index(pre,tar)
        
        return tdice

def main():
    image_path = './image'
    mask_path = './target'
    pred_path = './Deeplabv3+'
    # Unet AG-PCVNet Trans-AG-PCVNet TransUnet Deeplabv3+
    # MT PS-MT Cross_contraining Deep_cotraining
    # Dual-MT Siamese-MT
    # target
    dirs=os.listdir(image_path)
    dirs.sort()

    for file in dirs:
        m_path = os.path.join(mask_path, file)
        p_path = os.path.join(pred_path, file)

        print(m_path)
        print(p_path)

        print(f'name: {file} DSC = {count_index(p_path,m_path)}')



if __name__ == "__main__":
    main()