import os
import random
import cv2
import numpy as np
import torch
from torch.utils import data
import torchvision

class Dataset(data.Dataset):
    def __init__(self, mode, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.train_dataset = torchvision.datasets.MNIST(
            root='./',
            train= (mode == 'train'),
            download=True
        )
        self.backgroundImages = self.__init_background_images__('./net-images')
        self.patch_size = 28
        
    # 初始化背景图像列表
    def __init_background_images__(self, root):
        backgroundImages = []
        for name in os.listdir(root):
            imagePath = os.path.join(root, name)
            image = cv2.imread(imagePath)
            backgroundImages.append(image)
        return backgroundImages

    # 随机获取背景
    def __get_random_background__(self):
        image = random.choice(self.backgroundImages)
        height, width, _ = image.shape
        patch_size = self.patch_size
        xmin = random.randint(0, width - patch_size)
        ymin = random.randint(0, height - patch_size)
        background = image[ymin:ymin+patch_size, xmin:xmin+patch_size]
        return background
    
    def __getitem__(self, index):
        num_image, num_label = self.train_dataset[index]
        patch_size = self.patch_size
        
        # 数字
        foreground = np.array(num_image)
        foreground = foreground.reshape(patch_size, patch_size, 1)
        
        # 二值化
        _, mask = cv2.threshold(foreground, 130, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # cv2.imshow('mask',mask_inv)

        # 使用mask_inv给背景抠图
        background = self.__get_random_background__()
        background = cv2.bitwise_and(background, background, mask=mask_inv)
        # cv2.imshow('background', background)

        # 使用mask给前景图像抠图
        foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2RGB)
        foreground = cv2.bitwise_and(foreground, foreground, mask=mask)
        # cv2.imshow('foreground', foreground)

        # 前后景叠加
        blend_image = cv2.add(background, foreground)
        blend_image = blend_image.reshape(-1, patch_size, patch_size)
        
        label = mask
        if self.num_classes == 2:
            label[label==255] = 1
        elif self.num_classes == 11:
            label[label==0] = 10
            label[label==255] = num_label

        return blend_image.astype(np.float32), label.astype(np.int)
    def __len__(self):
        return len(self.train_dataset)

from config import get_args
if __name__ == '__main__':
    opt = get_args()
    dataset = Dataset(mode='train', num_classes=opt.num_classes)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    iter_loader = iter(loader)
    d = next(iter_loader)

    model = torch.load('model.pt', map_location='cpu')
    num_classes = model.num_classes
    outputs = model(d[0])
    preds = torch.argmax(outputs.data, 1)
    
    # print(preds.sum())
    if num_classes == 2:
        preds[preds==1] = 255
    elif num_classes == 11:
        preds[preds==10] = -1
        preds[preds==d[1]] = 255
        preds[preds==-1] = 0
    cv2.imwrite('./images/image.jpg', np.array(d[0].reshape(28, 28, 3)))
    cv2.imwrite('./images/pred.jpg', np.array(preds.reshape(28, 28, 1)))