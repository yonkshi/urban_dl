# full assembly of the sub-parts to form the complete net
from collections import OrderedDict

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, cfg):

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS

        if cfg.MODEL.BLOCK_ACTIVATION == 'PReLU':
            activation = nn.PReLU()
        else:
            activation = nn.ReLU(inplace=True)


        if cfg.MODEL.BLOCK_TYPE == 'Double':
            conv_block = double_conv
        elif cfg.MODEL.BLOCK_TYPE == 'Triple':
            conv_block = triple_conv

        self._cfg = cfg

        super(UNet, self).__init__()

        first_chan = cfg.MODEL.TOPOLOGY[0]
        self.inc = inconv(n_channels, first_chan, conv_block, activation)
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
            pooling_layer = self._build_pooling_layer(in_dim)
            layer = down(in_dim, out_dim, conv_block, activation, pooling_layer)

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

            layer = up_block(in_dim, out_dim, conv_block, activation, bilinear=cfg.MODEL.SIMPLE_INTERPOLATION)

            print(f'up{idx+1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx+1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, x):
        x1 = self.inc(x)

        inputs = [x1]
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
            x1 = layer(x1, x2)  # x1 for next up layer

        out = self.outc(x1)

        return out
    def _build_pooling_layer(self, in_channel):

        if self._cfg.MODEL.POOLING_TYPE == 'MaxPooling':
            pooling_layer = nn.MaxPool2d(2)
        elif self._cfg.MODEL.POOLING_TYPE == 'AvgPooling':
            pooling_layer = nn.AvgPool2d(2)
        elif self._cfg.MODEL.POOLING_TYPE == '2Stride':
            pooling_layer = nn.Conv2d(in_channel, in_channel, 2, 2)
        return pooling_layer

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