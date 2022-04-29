import torch
from torch import nn, optim
import torch.nn.functional as F


class autoencoder(nn.Module):
    def __init__(self, dim=256):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, dim), nn.Sigmoid())

        self.dim = dim

    def forward(self, x):
        # print(x.shape)

        x = self.encoder(x)
        # print(x.shape)

        x = self.decoder(x)
        # print(x.shape)
        return x


class convautoencoder(nn.Module):
    def __init__(self, ch=4):
        super(convautoencoder, self).__init__()

        #encoder
        self.conv1 = nn.Sequential(nn.Conv1d(1, ch, 3, padding=1), nn.ReLU())# (250*64)
        self.pool1 = nn.Sequential(nn.MaxPool1d(2,stride=2)) # （50*64）

        self.conv2 = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU()) # （50*64）
        self.pool2 = nn.Sequential(nn.MaxPool1d(2,stride=2)) # （10*64）

        self.conv3 = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())# （10*32)
        self.pool3 = nn.Sequential(nn.MaxPool1d(2,stride=2)) # （2*32)

        #decoder
        self.up1 = nn.Sequential(nn.ConvTranspose1d(ch, ch, 2, stride=2))# （10*32）
        self.conv4 = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())# （10*32）
        self.up2 = nn.Sequential(nn.ConvTranspose1d(ch, ch, 2, stride=2))# （50*64）
        self.conv5 = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())# （50*64）
        self.up3 = nn.Sequential(nn.ConvTranspose1d(ch, ch, 2, stride=2))# (250*64)
        self.conv6 = nn.Sequential(nn.Conv1d(ch, 1, 3, padding=1), nn.Sigmoid())# (250*1)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        # print('x1',x1.shape)
        x2 = self.pool1(x1)
        # print('x2',x2.shape)

        x3 = self.conv2(x2)
        # print('x3',x3.shape)
        x4 = self.pool2(x3)
        # print('x4',x4.shape)

        x5 = self.conv3(x4)
        # print('x5',x5.shape)
        x6 = self.pool3(x5)
        # print('x6',x6.shape)

        # decoder
        x7 = self.up1(x6)
        # print('x7',x7.shape)
        # merge7 = torch.cat([x7, x5], dim=1)
        # print('merge7',merge7.shape)
        # x8 = self.conv4(merge7)
        # print('x7',x7.shape)
        x8 = self.up2(x7)
        # print('x9', x9.shape)
        # merge9 = torch.cat([x9, x3], dim=1)
        # print('merge9 ', merge9 .shape)
        # x10 = self.conv5(merge9)
        # print('x8', x8.shape)
        x9 = self.up3(x8)
        # print('x9', x9.shape)
        x10 = self.conv6(x9)
        # print('x10', x10.shape)
        return x10

if __name__ == '__main__':
    with torch.no_grad():
        # net = autoencoder().cuda()

        net = convautoencoder().cuda()
 
        x = torch.randn(64, 1, 256).cuda()
        y = net(x)
        assert x.shape == y.shape