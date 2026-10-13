""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.tranf4 = DnCNN(1, 512, 3)
        self.tranf3 = DnCNN(1, 256, 3)
        # self.up3 = Net(num_channels=n_classes, base_filter=64, feat=256, num_stages=10, scale_factor=4)  # DBPN
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)

    # 超分不用DBPN用上采
    def forward(self, x_src, x_tar, LL2, LL3):
        x_src = self.inc(x_src)
        x1_src = x_src
        x_src = self.down1(x_src)
        x2_src = x_src
        x_src = self.down2(x_src)
        # x3_src = x_src
        x_src = self.down3(x_src)
        # x4_src = x_src
        x_src = self.down4(x_src)
        x5_src = x_src
        x_tar = self.inc(x_tar)
        x_tar = self.down1(x_tar)
        x_tar = self.down2(x_tar)
        x_tar = self.down3(x_tar)
        x_tar = self.down4(x_tar)
        x5_tar = x_tar
        x5_transf = self.tranf4(LL3)
        x4_transf = self.tranf3(LL2)
        x_src = self.up1(x_src, x5_transf)
        x_src = self.up2(x_src, x4_transf)
        x_src = self.up3(x_src, x2_src)
        x_src = self.up4(x_src, x1_src)
        logits = self.outc(x_src)
        return logits, x5_src, x5_src
