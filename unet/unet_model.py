# full assembly of the sub-parts to form the complete net
from collections import OrderedDict

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, cfg):

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS

        super(UNet, self).__init__()
        first_chan = cfg.MODEL.TOPOGRAPHY[0]
        self.inc = inconv(n_channels, first_chan)

        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 256)
        # self.down4 = down(256, 256)
        # self.up1 = up(512, 256)
        # self.up2 = up(512, 128)
        # self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        self.outc = outconv(first_chan, n_classes)

        # Variable scale

        down_topo = cfg.MODEL.TOPOGRAPHY
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan] # topography upwards
        up_dict = OrderedDict()


        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers-1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx+1] if is_not_last_layer else down_topo[idx] # last layer
            layer = down(in_dim, out_dim)
            print(f'down{idx+1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx+1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)


        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = up(in_dim, out_dim, bilinear=cfg.MODEL.SIMPLE_INTERPOLATION)

            print(f'up{idx+1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx+1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

        # self.out_softmax = nn.Softmax2d()

    def forward(self, x):
        x1 = self.inc(x)
        inputs = [x1]
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)

        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        # Upward U:
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            is_first_layer = idx == 0
            x2 = inputs[idx]
            x1 = layer(x1, x2) # x1 for next up layer

        out = self.outc(x1)
        # out = self.out_softmax(out)

        return out
