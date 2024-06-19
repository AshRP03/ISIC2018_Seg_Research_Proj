import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.center = self.conv_block(512, 1024)

        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        c = self.center(F.max_pool2d(e4, 2))

        d4 = self.decoder4(F.interpolate(c, scale_factor=2, mode='bilinear', align_corners=True))
        d3 = self.decoder3(F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True))
        d2 = self.decoder2(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True))
        d1 = self.decoder1(F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True))

        out = self.final(d1)
        return out

model = UNet()