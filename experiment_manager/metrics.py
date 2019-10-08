import torch


def true_pos(y_true, y_pred):
    return torch.sum(y_true * torch.round(y_pred))


def false_pos(y_true, y_pred):
    return torch.sum(y_true * (1. - torch.round(y_pred)))


def false_neg(y_true, y_pred):
    return torch.sum((1. - y_true) * torch.round(y_pred))


def precision(y_true, y_pred):
    return true_pos(y_true, y_pred) / \
           (true_pos(y_true, y_pred) + false_pos(y_true, y_pred))


def recall(y_true, y_pred):
    return true_pos(y_true, y_pred) / \
           (true_pos(y_true, y_pred) + false_neg(y_true, y_pred))

def f1_score(preds:torch.Tensor, gts:torch.Tensor):
    # TODO Add support for batch
    gts = gts.float()
    preds = preds.float()

    with torch.no_grad():
        recall_val = recall(gts, preds)
        precision_val = precision(gts, preds)
        f1 = 2. * recall_val * precision_val / (recall_val + precision_val)

    return f1


