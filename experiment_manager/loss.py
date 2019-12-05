import torch
def soft_dice_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))