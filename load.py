import torch
import torch.nn as nn
import cv2
import numpy as np
import os

class DDSRCNNNet(torch.nn.Module):
    def __init__(self):
        super(DDSRCNNNet, self).__init__()
        # 输入是 256 x 256
        self.conv1 = nn.Sequential(  # 输入 256 x 256 * 1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # (256-3+2)/1+1 = 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # (256-3+2)/1+1 = 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (256-2)/2+1 = 128
        self.conv2 = nn.Sequential(  # 输入 128 x 128 * 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (128-3+2)/1+1 = 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # (128-3+2)/1+1 = 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (128-2)/2+1 = 64
        self.conv3 = nn.Sequential(  # 输入 64 x 64 * 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (64-3+2)/1+1 = 64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            # nn.Upsample(scale_factor=2), # 128x128x256
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),  # (128-3+2)/1+1 = 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # (128-3+2)/1+1 = 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # add dec3 + conv2
        self.upSample = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            # nn.Upsample(scale_factor=2),  # 256x256x128
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),  # (256-3+2)/1+1 = 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # (256-3+2)/1+1 = 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # add upSample conv1
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        enc1 = self.conv1(x)
        down1 = self.maxpool1(enc1)
        enc2 = self.conv2(down1)
        down2 = self.maxpool2(enc2)
        enc3 = self.conv3(down2)
        dec3 = self.dec3(enc3)
        add2 = dec3 + enc2
        dec2 = self.upSample(add2)
        add1 = dec2 + enc1
        dec1 = self.dec1(add1)
        print(dec1.shape)
        return dec1


import math


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    return 20 * math.log10(1 / math.sqrt(mse))

''''''''''
if __name__ == '__main__':
    device = torch.device("cuda")
    model = torch.load('checkpoints_simu/dsrcnn_model_285.pt')
    model.to(device)
    model.eval()
    pic_path = r'rectLR.png'

    imgA = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(r'rectGT.png', cv2.IMREAD_GRAYSCALE)

    imgA = np.expand_dims(imgA, axis=0)
    imgA = np.expand_dims(imgA, axis=0)
    print(imgA.shape)
    imgA = torch.Tensor(imgA).to(device)
    output = model(imgA)
    output = output.reshape(256, 256)
    output = output.cpu().detach().numpy().copy()
    output = output.astype(np.uint8)
    cv2.imshow('s', output)
    cv2.waitKey(0)

    imgout_path = r''
    cv2.imwrite(imgout_path, output)

    print(psnr(label, output))
'''''''''
if __name__ == '__main__':
    device = torch.device("cuda")
    model = torch.load('out/model/dsrcnn_model_30_256.pt')
    model.to(device)
    model.eval()
    pic_path = r'out/test_img_input/'
    imgout_path = r'out/test_img_output/'

    if not os.path.exists(imgout_path):
        os.makedirs(imgout_path)
    image_list = os.listdir(pic_path)
    i = 0
    for file in image_list:
        i = i + 1
        imgA = cv2.imread(pic_path + file, cv2.IMREAD_GRAYSCALE)
        # imgA = cv2.resize(src=imgA, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)#####这一句是512*512才加的
#    label = cv2.imread(r'rectGT.png', cv2.IMREAD_GRAYSCALE)

        imgA = np.expand_dims(imgA, axis=0)
        imgA = np.expand_dims(imgA, axis=0)
        print(imgA.shape)
        imgA = torch.Tensor(imgA).to(device)
        output = model(imgA)
        output = output.reshape(256, 256)
        # output = output.reshape(512, 512)
        output = output.cpu().detach().numpy().copy()
        output = output.astype(np.uint8)
        output = cv2.medianBlur(output, 5)######加入了opencv的中值滤波，将图片中250以上的白点值用周围的点平均值代替
        cv2.imshow('s', output)
        cv2.waitKey(0)

        imgout_path = r'breast_att/'
       # cv2.imwrite(imgout_path+ file, output)
        cv2.imwrite('breast_att.png', output)
print('over')
#    print(psnr(label, output))

