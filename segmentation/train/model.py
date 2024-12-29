import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# UNet 모델 정의
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 384)
        
        # 더 높은 채널로 설정
        self.bottleneck = self.conv_block(384, 512)
        
        # 디코더
        self.decoder4 = self.conv_block(512 + 384, 256)
        self.decoder3 = self.conv_block(256 + 256, 128)
        self.decoder2 = self.conv_block(128 + 128, 64)
        self.decoder1 = nn.Conv2d(64 + 64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))

        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        
        # 디코딩 경로
        dec4 = self.decoder4(torch.cat([nn.Upsample(scale_factor=2)(bottleneck), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([nn.Upsample(scale_factor=2)(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([nn.Upsample(scale_factor=2)(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([nn.Upsample(scale_factor=2)(dec2), enc1], dim=1))
        
        return dec1