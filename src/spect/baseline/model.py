# src/spect/baseline/model.py

#--------------------------------------------------------------------------
# Copied verbatim from Wei Miao's original implementation
# (https://github.com/ucapwmi/SPECT_codes, models/Unet/model.py):
# 3D U-Net baseline: 4-level encoder (32->64->128->256), bottleneck (512), symmetric decoder with skip connections, 1x1x1 output conv.
#--------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2) if dropout else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
        )
        self.conv = ConvBlock(in_channels=out_channels * 2, out_channels=out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CustomUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)

        # bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, dropout=True)

        # decoder
        self.up4 = UpBlock(base_channels * 16, base_channels * 8, dropout=True)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, dropout=True)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels)

        # final conv
        self.final = nn.Conv3d(base_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.final(d1)



# --------------------------------------------------------------------------
# Swin UNETR baseline copied verbatim from Wei Miao's original implementation
# (https://github.com/ucapwmi/SPECT_codes, models/Swin_UNETR/model.py).
# Uses MONAI's built-in SwinUNETR directly (no custom architecture). Merged
# into this single model.py alongside CustomUNet3D above, rather than kept
# as two separate files as in his repo, so this package has one file per
# category (model, dataset, quantification, ...) instead of one per
# architecture.
# --------------------------------------------------------------------------
def get_swin_unetr(device, roi_size=(128, 128, 128)):
    """Build SwinUNETR model. See config.py SWIN_UNETR_MODEL_CONFIG for
    the architecture parameters documented alongside the other configs."""
    model = SwinUNETR(
        img_size=roi_size,
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=True,
        use_v2=True,
    ).to(device)
    return model