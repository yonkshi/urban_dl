# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class triple_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(triple_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class attention_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.W2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.MaxPool2d(2)
        )

        self.W1 = nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        self.psi =  nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2,): # X1 is upstream, X2 is skip


        xl_size_orig = x2.size()
        xl_ = self.W2(x2)
        x1 = self.W1(x1)

        psi = self.psi(xl_ + x1)

        upsampled_psi = F.interpolate(psi, size=xl_size_orig[2:], mode='bilinear', align_corners=False)

        # scale features with attention
        attention = upsampled_psi.expand_as(x2)

        return attention * x2


class ContextLayer(nn.Module):
    def __init__(self, channels, dilation, include_activation = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        if include_activation:
            self.conv.add_module('relu', nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(inconv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )


    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block, bilinear=True, ):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class attention_up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block, bilinear=True, ):
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.attention = attention_block(in_ch, out_ch)
        self.conv = conv_block(in_ch, out_ch)
        print('in', in_ch, 'out', out_ch)

    def forward(self, x1, x2):
        x2 = self.attention(x1, x2)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
