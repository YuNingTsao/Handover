import cv2
import numpy as np
import os

def apply_mask_with_transparency(image_path, mask_path, alpha=0.5, color=(0, 255, 0)):
    # 创建一个彩色的掩码图像
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_AREA)

    # 叠加彩色掩码到原图上
    overlay = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
    return overlay

def main():
    # 读取原始图像和分割掩码
    image_path = './image'
    # 0132_09_1 003000-60S-004_1 2933301-5_OD_3
    mask_path = './Siamese-MT'
    # Unet AG-PCVNet Trans-AG-PCVNet TransUnet Deeplabv3+
    # MT PS-MT Cross_contraining Deep_cotraining
    # Dual-MT Siamese-MT
    # target

    dirs=os.listdir(image_path)
    dirs.sort()

    for file in dirs:
        img_path = os.path.join(image_path, file)
        m_path = os.path.join(mask_path, 'RGB_'+file)
        # 设置透明度和颜色
        alpha = 0.5 # 透明度
        color = (0, 255, 0) # 掩码颜色，绿色
        # 应用透明掩码
        result = apply_mask_with_transparency(img_path, m_path, alpha, color)
        # 显示结果
        # cv2.imshow('Overlay', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 保存结果
        result_path = mask_path+'/overlap_'+file
        cv2.imwrite(result_path, result)

if __name__ == "__main__":
    main()