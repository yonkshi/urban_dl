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
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps

    return 1 - ((2. * intersection) / denom)
def jaccard_like_balanced_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps
    piccard = (2. * intersection)/denom

    n_iflat = 1-iflat
    n_tflat = 1-tflat
    neg_intersection = (n_iflat * n_tflat).sum()
    neg_denom = (n_iflat**2 + n_tflat**2).sum() - (n_iflat * n_tflat).sum()
    n_piccard = (2. * neg_intersection)/neg_denom

    return 1 - piccard - n_piccard

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