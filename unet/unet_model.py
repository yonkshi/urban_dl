# full assembly of the sub-parts to form the complete net
from collections import OrderedDict

import torch.nn.functional as F
import numpy as np

from .unet_parts import *
from .dpn import *

class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS

        if cfg.MODEL.BLOCK_ACTIVATION == 'PReLU':
            self.activation  = nn.PReLU()
        else:
            self.activation = nn.ReLU(inplace=True)


        if cfg.MODEL.BLOCK_TYPE == 'Double':
            conv_block = double_conv
        elif cfg.MODEL.BLOCK_TYPE == 'Triple':
            conv_block = triple_conv

        self._cfg = cfg



        first_chan = cfg.MODEL.TOPOLOGY[0]
        self.inc = inconv(n_channels, first_chan, conv_block, self.activation)
        self.outc = outconv(first_chan, n_classes)
        self.multiscale_context_enabled = cfg.MODEL.MULTISCALE_CONTEXT.ENABLED
        self.multiscale_context_type = cfg.MODEL.MULTISCALE_CONTEXT.TYPE

        up_block = attention_up if cfg.MODEL.USE_ATTENTION else up

        # Variable scale
        down_topo = cfg.MODEL.TOPOLOGY
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan] # topography upwards
        up_dict = OrderedDict()

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers-1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx+1] if is_not_last_layer else down_topo[idx] # last layer
            layer = down(in_dim, out_dim, conv_block, self.activation, self._cfg.MODEL.POOLING_TYPE)

            print(f'down{idx+1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx+1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)
        bottleneck_dim = out_dim

        # context layer
        if self.multiscale_context_enabled:
            self.multiscale_context = MultiScaleContextForUNet(cfg, bottleneck_dim)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]

            layer = up_block(in_dim, out_dim, conv_block, self.activation, bilinear=cfg.MODEL.SIMPLE_INTERPOLATION)
            if idx == 0 and cfg.MODEL.LAST_LAYER_RESIDUAL:
                in_dim = up_topo[x1_idx]
                layer = residual_up(in_dim, out_dim, conv_block, self.activation, bilinear=cfg.MODEL.SIMPLE_INTERPOLATION)

            print(f'up{idx+1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx+1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, x):
        inc_out = self.inc(x)

        inputs = [inc_out]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        #Multiscale context
        if self.multiscale_context_enabled:
            bottleneck_features = inputs.pop()
            context = self.multiscale_context(bottleneck_features)
            inputs.append(context)

        # Upward U:
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x_out = layer(x1, x2)  # x1 for next up layer
            x1 = x_out

        out = self.outc(x1)

        return out

class MultiScaleContextForUNet(nn.Module):
    def __init__(self, cfg, bottlneck_dim):
        super().__init__()
        self._cfg = cfg
        self.multiscale_context_type = cfg.MODEL.MULTISCALE_CONTEXT.TYPE
        self.context = self.build_multiscale_context(bottlneck_dim)

    def build_multiscale_context(self, bottleneck_dim):
        context_layers = []
        for i, layer_dilation in enumerate(self._cfg.MODEL.MULTISCALE_CONTEXT.DILATION_TOPOLOGY):
            layer = ContextLayer(bottleneck_dim, layer_dilation)
            context_layers.append(layer)
        if self.multiscale_context_type == 'Simple':
            context = nn.Sequential(*context_layers)
        if self.multiscale_context_type == 'PyramidSum':
            context =  nn.ModuleList(context_layers)
        if self.multiscale_context_type == 'ParallelSum':
            context =  nn.ModuleList(context_layers)
        return context

    def forward(self, x):
        if self.multiscale_context_type == 'Simple':
            context = self.context(x)
        elif self.multiscale_context_type == 'PyramidSum':
            context = x
            for layer in self.context:
                context = layer(context)
        elif self.multiscale_context_type == 'ParallelSum':
            context = x
            for layer in self.context:
                context += layer(x)

        return context

# ===== Victor's implementation

class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, channels, reduction=16, concat=False):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return chn_se + spa_se

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class Dpn92_Unet_Double(nn.Module):
    def __init__(self, pretrained='imagenet+5k', **kwargs):
        super(Dpn92_Unet_Double, self).__init__()

        encoder_filters = [64, 336, 704, 1552, 2688]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1]),
                                     SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2]),
                                     SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3]),
                                     SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4]),
                                     SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2, decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = dpn92(pretrained=pretrained)

        # conv1_new = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # _w = encoder.blocks['conv1_1'].conv.state_dict()
        # _w['weight'] = torch.cat([0.5 * _w['weight'], 0.5 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)

        self.conv1 = nn.Sequential(
            encoder.blocks['conv1_1'].conv,  # conv
            encoder.blocks['conv1_1'].bn,  # bn
            encoder.blocks['conv1_1'].act,  # relu
        )
        self.conv2 = nn.Sequential(
            encoder.blocks['conv1_1'].pool,  # maxpool
            *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
        )
        self.conv3 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        self.conv4 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        self.conv5 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    def forward1(self, x):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        enc1 = (torch.cat(enc1, dim=1) if isinstance(enc1, tuple) else enc1)
        enc2 = (torch.cat(enc2, dim=1) if isinstance(enc2, tuple) else enc2)
        enc3 = (torch.cat(enc3, dim=1) if isinstance(enc3, tuple) else enc3)
        enc4 = (torch.cat(enc4, dim=1) if isinstance(enc4, tuple) else enc4)
        enc5 = (torch.cat(enc5, dim=1) if isinstance(enc5, tuple) else enc5)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9,
                                       enc1], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10

    def forward(self, x):

        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()