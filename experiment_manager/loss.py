import torch
def soft_dice_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

def jaccard_like_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum()

    return 1 - ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

def soft_dice_loss_balanced(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    dice_pos = ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

    negatiev_intersection = ((1-iflat) * (1 - tflat)).sum()
    dice_neg =  (2 * negatiev_intersection) / ((1-iflat).sum() + (1-tflat).sum() + eps)

    return 1 - dice_pos - dice_neg