import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50
from torch.autograd import Function
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

        modules = [nn.Linear(2048, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, 1),]

        if cfg.MODEL.ADVERSARIAL_REFINEMENT.ENABLE_GRADIENT_REVERSAL:
            modules = [GradientReversal()] + modules

        self.output = nn.Sequential( *modules)

    def forward(self, x):
        layer0 = self.input(x)
        feature_out = self.backbone(layer0).squeeze()

        y = self.output(feature_out)
        return y

class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input

class GradientReversal(torch.nn.Module):
    def forward(self, x):
        return RevGrad.apply(x)