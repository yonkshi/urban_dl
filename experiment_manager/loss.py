import torch
def soft_dice_loss(input:torch.Tensor, target:torch.Tensor):
    smooth = 1.

    iflat = input.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))