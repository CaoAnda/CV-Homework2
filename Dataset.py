import os
import random
import cv2
import numpy as np
from torch.utils import data
import torchvision
from torchvision.transforms import transforms

class TrainDataset(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.train_dataset = torchvision.datasets.MNIST(
            root='./',
            train=True,
            # transform=transform,
            download=True
        )
        self.backgroundImages = self.__init_background_images__('./net-images')
        self.patch_size = 28
        # self.toTensor = transforms.ToTensor()
        
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
        
        label = mask
        label[label==255] = 1
        # cv2.imshow('blend', blend_image)

        return blend_image, label
    def __len__(self):
        return len(self.train_dataset)

if __name__ == '__main__':
    dataset = TrainDataset()
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    iter_loader = iter(loader)
    d = next(iter_loader)
    d = next(iter_loader)
    cv2.imwrite('he.jpg', np.array(d[0].reshape(28, 28, 3)))
    cv2.imwrite('label.jpg', np.array(d[1].reshape(28, 28, 1)))
    # cv2.waitKey()
    pass

