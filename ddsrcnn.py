import pickle
import torch
import torch.nn as nn
import time
import glob
import numpy as np
import cv2
from skimage.io import imread, imshow


train_orgs = sorted(glob.glob('data/256px/teacher_data_train/*.png'))
train_lows = sorted(glob.glob('data/256px/learning_data_train/*.png'))
test_orgs = sorted(glob.glob('data/256px/teacher_data_eval/*.png'))
test_lows = sorted(glob.glob('data/256px/learning_data_eval/*.png'))
PATCH_SIZE = 256   ##修改创建空白数据集的图片大小，这个图片大小给定才可以赋值进去，在下面20-23行中

# 从文件中读取数据
def readFile():
    # 获取测试集
    X_train = np.zeros((len(train_lows), PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32)
    Y_train = np.zeros((len(train_orgs), PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32)
    X_test = np.zeros((len(test_lows), PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32)
    Y_test = np.zeros((len(test_orgs), PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32)
    for n in range(len(train_orgs)):
        low_train = imread(train_lows[n], as_gray=True)
        low_train = cv2.resize(src=low_train, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)###将图像放大，用opencv的插值法
        # low_train = low_train.resize(PATCH_SIZE, PATCH_SIZE)
        X_train[n] = np.expand_dims(low_train, axis=2)
        org_train = imread(train_orgs[n], as_gray=True)
        org_train = cv2.resize(src=org_train, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        Y_train[n] = np.expand_dims(org_train, axis=2)
    for n in range(len(test_orgs)):
        low_test = imread(test_lows[n], as_gray=True)
        low_test = cv2.resize(src=low_test, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        X_test[n] = np.expand_dims(low_test, axis=2)
        org_test = imread(test_orgs[n], as_gray=True)
        org_test = cv2.resize(src=org_test, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        Y_test[n] = np.expand_dims(org_test, axis=2)
    X_train = X_train.reshape((len(train_lows), 1, PATCH_SIZE, PATCH_SIZE))
    Y_train = Y_train.reshape((len(train_orgs), 1, PATCH_SIZE, PATCH_SIZE))
    X_test = X_test.reshape((len(test_lows), 1, PATCH_SIZE, PATCH_SIZE))
    Y_test = Y_test.reshape((len(test_orgs), 1, PATCH_SIZE, PATCH_SIZE))
    return X_train, Y_train, X_test, Y_test


# 定义getDataset类，继承Dataset方法，重写__getitem__()和__len__()方法
class getDataset(torch.utils.data.Dataset):
    # 初始化函数，得到图像和标签
    def __init__(self, data, label):
        self.data = data
        self.label = label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分
    def __len__(self):
        return len(self.data)


def PSNRLoss(y_true, y_pred):
    return -10. * torch.log(torch.mean(torch.square(y_pred - y_true)))


n_epochs = 30   # epoch的数目
learning_rate = 0.0001  # 学习率
lossfunc = torch.nn.MSELoss()  #
# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")
# 获取数据
train_data, train_label, test_data, test_label = readFile()
# 创建Dataset对象
train_dataset = getDataset(train_data, train_label)
test_dataset = getDataset(test_data, test_label)
import math
def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    return 20 * math.log10(1 / math.sqrt(mse))

# DDSRCNNNet网络
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
            nn.Linear(PATCH_SIZE, PATCH_SIZE),
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
        return dec1


# 训练神经网络
def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 开始训练
    print(optimizer)
    for epoch in range(n_epochs):
        batch_size = 4
        # 创建加载器
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        train_loss = 0.0
        for data, target in train_loader:
            # cuda加速
            data = data.to(device)
            target = target.to(device)
            # 将梯度初始化为0
            optimizer.zero_grad()
            output = model(data)
            # 计算loss标量，用于反向传播和绘图
            loss = lossfunc(output, target).to(device)
            # 反向传播求梯度
            loss.backward()
            # 更新所有参数
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Batch: ' + str(batch_size) + ' lr=' + str(learning_rate)
              + ' Epoch: {}  Training Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        test_loss = test(model, test_loader)
        if np.mod(epoch, 5) == 0:
            torch.save(model, 'out/model/dsrcnn_model_{}.pt'.format(epoch))


# 在数据集上测试神经网络
def test(model, test_loader):
    test_loss = 0.0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            # cuda加速
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = model(images)
            # 计算测试loss
            loss = lossfunc(outputs, labels).to(device)
            test_loss += loss.item() * images.size(0)
    test_loss = test_loss / len(test_loader.dataset)
    print("Test_loss:" + str(test_loss))
    return test_loss


model = DDSRCNNNet().to(device)
start = time.perf_counter()
train()
end = time.perf_counter()
print('Each epoch need %.10f seconds' % ((end - start)/n_epochs))
print("finished!")

