from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import time
import os
import json
from Dataset import Dataset

from config import get_args

opt = get_args()
# 设置随机种子
import random

from net import LeNet
seed = opt.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    train_data = Dataset(mode='train')
    test_data = Dataset(mode='test')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = opt.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size)

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    epoch_num = opt.epochs

    model = LeNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    train_loss = []
    val_loss = []
    train_epochs_loss = []
    val_epochs_loss = []
    val_epochs_acc = []

    now_time = time.strftime('%m-%d_%H-%M-%S', time.localtime())
    log_dir = './logs/%s'%now_time
    os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'config'), 'w+') as file:
        json.dump(opt.__dict__, file, indent=4)

    for epoch in range(epoch_num):
        model.train()
        for images, labels in tqdm(train_loader, desc='epoch {}'.format(epoch)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        train_epochs_loss.append(np.average(train_loss))
        print('train loss:{}'.format(np.average(train_loss)))
        
        model.eval()
        # with torch.no_grad():
        #     correct = 0
        #     total = 0
        #     for images, labels in tqdm(test_loader, desc='test'):
        #         images = images.to(device)
        #         labels = labels.to(device)

        #         outputs = model(images)
        #         loss = criterion(outputs, labels)
        #         preds = torch.argmax(outputs.data, 1)

        #         val_loss.append(loss.item())
        #         total += labels.size(0)
        #         correct += (preds == labels).sum().item()
        #     val_epochs_loss.append(np.average(val_loss))
        #     val_epochs_acc.append(np.average(correct / total))
        #     print('loss:{}\tacc:{}%'.format(np.average(val_loss), 100 * correct / total))
        
        print()

        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.plot(val_epochs_acc, '-s')
        plt.title("val_epochs_acc")
        plt.subplot(122)
        plt.plot(train_epochs_loss, '-o', label="train_loss")
        plt.plot(val_epochs_loss, '-o', label="valid_loss")
        plt.title("epochs_loss")
        plt.legend()
        plt.savefig(os.path.join(log_dir,'analys.png'))
        plt.show()
        plt.close()

        with open(os.path.join(log_dir, 'log.txt'), 'a+') as file:
            file.write('\t'.join([str(val_epochs_acc[-1]), str(train_epochs_loss[-1]), str(val_epochs_loss[-1])]))
            file.write('\n')