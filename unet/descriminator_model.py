import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50
import torchvision


class RefinementDescriminator(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, cfg, ):
        super().__init__()
        original_resnet = torchvision.models.resnet50()
        self.input = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone = nn.Sequential(
            *list(original_resnet.children())[1:-1]
        )
        self.output = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1),
        )

    def forward(self, x):
        layer0 = self.input(x)
        feature_out = self.backbone(layer0).squeeze()

        y = self.output(feature_out)
        return y
