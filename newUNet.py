from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose
)
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import datetime

aug = Compose([VerticalFlip(p=0.5),
               HorizontalFlip(p=0.5)])


class MyDataset(Dataset):

    def __init__(self,
                 transform=None,
                 mode='train',
                 img_path='BBBC018_v1_images-fixed',
                 mask_path='BBBC018_v1_outlines'):
        self.mode = mode
        self.transform = transform
        self.imgs = [os.path.join(img_path, self.mode, i) for i in os.listdir(os.path.join(img_path, self.mode))]
        if mode != 'test':
            self.masks = [os.path.join(mask_path, self.mode, i) for i in os.listdir(os.path.join(mask_path, self.mode))]
            self.masks.sort()
        self.imgs.sort()

    def __getitem__(self, i):
        seed = np.random.randint(42)
        # print(self.imgs[i])
        img = cv2.imread(self.imgs[i])
        if self.mode != 'test':
            mask = cv2.imread(self.masks[i], 0)

        if self.mode == 'train':
            augmented = aug(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)) / 255.

        if self.mode == 'test':
            img = torch.from_numpy(img).float()
            return img, []
        # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        mask = np.array(mask, dtype=np.int64)
        mask = np.where(mask > 0, 1, 0)

        img = torch.from_numpy(img).float()
        return img, mask

    def __len__(self):
        return len(self.imgs)

    def get_filename(self, i):
        return self.imgs[i].split('/')[-1]


trainloader = DataLoader(MyDataset(mode='train'),
                         batch_size=1,
                         shuffle=True, num_workers=1)

valloader = DataLoader(MyDataset(mode='val'),
                       batch_size=1,
                       shuffle=True, num_workers=1)


def calc_iou(prediction, ground_truth):
    n_images = len(prediction)
    intersection, union = 0, 0
    for i in range(n_images):
        intersection += np.logical_and(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()
        union += np.logical_or(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()
    return float(intersection) / union


class DecodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first=False):
        super(DecodeBlock, self).__init__()
        if first:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(),
                                      nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                      nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(),
                                      nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class EncodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncodeBlock, self).__init__()
        self.upconv = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(),
                                  nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU())

    def forward(self, curr, prev):
        curr = self.upconv(curr)
        diff = curr.size()[2] - prev.size()[2]
        prev = F.pad(prev, (diff // 2, diff - diff // 2, diff // 2, diff - diff // 2))
        x = torch.cat([prev, curr], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.first = DecodeBlock(n_channels, 64, first=True)
        self.down1 = DecodeBlock(64, 128)
        self.down2 = DecodeBlock(128, 256)
        self.down3 = DecodeBlock(256, 512)
        self.down4 = DecodeBlock(512, 1024)
        self.down5 = DecodeBlock(1024, 1024)

        self.up0 = EncodeBlock(2048, 512)
        self.up1 = EncodeBlock(1024, 256)
        self.up2 = EncodeBlock(512, 128)
        self.up3 = EncodeBlock(256, 64)
        self.up4 = EncodeBlock(128, 64)

        # self.down1 = DecodeBlock(64, 128, first=False)
        # self.down2 = DecodeBlock(128, 256, first=False)
        # self.down3 = DecodeBlock(256, 512, first=False)
        # self.down4 = DecodeBlock(512, 1024, first=False)
        # self.down5 = DecodeBlock(1024, 2048, first=False)
        # self.down6 = DecodeBlock(2048, 2048, first=False)
        # self.up00 = EncodeBlock(4096, 1024)
        # self.up0 = EncodeBlock(2048, 512)
        # self.up1 = EncodeBlock(1024, 256)
        # self.up2 = EncodeBlock(512, 128)
        # self.up3 = EncodeBlock(256, 64)
        # self.up4 = EncodeBlock(128, 64)

        self.last = nn.Conv2d(64, n_classes, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        # prevList = []
        x1 = self.first(x)
        # prevList.append(x)
        #
        # x = self.down1(x)
        # prevList.append(x)
        #
        # x = self.down2(x)
        # prevList.append(x)
        #
        # x = self.down3(x)
        # prevList.append(x)
        #
        # x = self.down4(x)
        # prevList.append(x)
        #
        # x = self.down5(x)
        #
        # x = self.up0(x, prevList[-1])
        # x = self.up1(x, prevList[-2])
        # x = self.up2(x, prevList[-3])
        # x = self.up3(x, prevList[-4])
        # x = self.up4(x, prevList[-5])

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.last(x)

        return F.sigmoid(x)


model = UNet(3, 2)
model.init_weights()
model.cuda()
writer = SummaryWriter()

batch_size = 1
num_workers = 1
num_epochs = 3
lr = 0.0001

criterion = nn.CrossEntropyLoss(torch.Tensor([1., 22.]).cuda())
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def run_epoch(model, dataloader, mode):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    loss = []
    acc = []
    for batch_index, data in enumerate(dataloader):
        inputs, gt = data
        outputs = model(inputs.cuda())
        loss_ = criterion(outputs, gt.cuda())
        if mode == 'train':
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
        loss.append(loss_.cpu().item())
        outputs = outputs.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        outputs = np.where(outputs[:, 1, :, :] > 0.5, 1, 0)
        acc.append(calc_iou(outputs, gt))
    return np.array(loss).mean(), np.array(acc).mean()


train_loss = []
train_accuracy = []
val_accuracy = []
val_loss = []
lr_list = []

for epoch in range(num_epochs):
    if epoch == 15:
        lr /= 10.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    if epoch == 35:
        lr /= 10.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    if epoch == 55:
        lr /= 10.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    a, l = run_epoch(model, trainloader, 'train')
    train_loss.append(a)
    train_accuracy.append(l)

    a, l = run_epoch(model, valloader, 'val')
    val_loss.append(a)
    val_accuracy.append(l)

    lr_list.append(lr)

    writer.add_scalar('data/train_loss', train_loss[-1], epoch)
    writer.add_scalar('data/train_accuracy', train_accuracy[-1], epoch)
    writer.add_scalar('data/loss', val_accuracy[-1], epoch)
    writer.add_scalar('data/lr', lr, epoch)

    print('Epoch {} -- Train Loss {:.2f}%, Train Acc {:.2f}%, Val Acc {:.2f}%'.format(
        epoch, train_loss[-1] * 100, train_accuracy[-1] * 100, val_accuracy[-1] * 100))

writer.export_scalars_to_json("data/scalars/all_scalars.json")
writer.close()
