import torch
from sklearn.metrics import roc_auc_score, roc_curve


class MultiThresholdMetric():
    def __init__(self, y_true, y_pred, threshold):

        # FIXME Does not operate properly

        '''
        Takes in rasterized and batched images
        :param y_true: [B, H, W]
        :param y_pred: [B, C, H, W]
        :param threshold: [Thresh]
        '''
        self._y_true = y_true
        self._y_pred = y_pred
        B, C, H, W = y_pred.shape
        self.numel = C * H * W # total number of pixels per image
        self._thresholds = threshold
        self._data_dims = (-1, -2, -3) # For a B/W image, it should be [B, ...,

        self._normalize_dimensions()
        self._build_threshold_for_computation()


    def _normalize_dimensions(self):
        ''' Converts y_truth, y_label and threshold to [B, Thres, C, H, W]'''
        # Naively assume that all of existing shapes of tensors, we transform [B, H, W] -> [B, Thresh, C, H, W]
        self._thresholds = self._thresholds[None, :, None, None, None] # [B, T, C]
        self._y_pred = self._y_pred[:, None, ...]  # [B, Thresh, C, ...]
        self._y_true = self._y_true[:, None, None, ...] # [B, Thresh, C, ...]

    def _build_threshold_for_computation(self):
        ''' Vectorize y_pred so that it contains N_THRESH aligned dimension'''
        self._y_pred = (self._y_pred - self._thresholds + 0.5).clamp(0, 1.0)

    @property
    def TP(self):
        if hasattr(self, '_TP'): return self._TP
        self._TP = (self._y_true * torch.round(self._y_pred)).sum(dim=self._data_dims)
        return self._TP
    @property
    def FP(self):
        if hasattr(self, '_FP'): return self._FP
        self._FP = (self._y_true * (1. - torch.round(self._y_pred))).sum(dim=self._data_dims)
        return self._FP

    @property
    def FN(self):
        if hasattr(self, '_FN'): return self._FN
        self._FN = ((1. - self._y_true) * torch.round(self._y_pred)).sum(dim=self._data_dims)
        return self._FN

    @property
    def precision(self):
        if hasattr(self, '_precision'):
            '''precision previously computed'''
            return self._precision

        denom = (self.TP + self.FP).clamp(10e-05)
        self._precision = self.TP / denom
        return self._precision

    @property
    def recall(self):
        if hasattr(self, '_recall'):
            '''recall previously computed'''
            return self._recall

        denom = (self.TP + self.FN).clamp(10e-05)
        self._recall = self.TP / denom
        return self._recall


    def compute_f1(self):
        denom = (self.precision + self.recall).clamp(10e-05)
        return 2 * self.precision * self.recall / denom



    def compute_roc_curve(self):
        # TODO Not quite right yet. Should be based on all possible thresholds, rather than a predefined thresholds
        fpr = self.FP / self.numel
        tpr = self.TP / self.numel
        sorted_fpr_idx = fpr.argsort()
        sorted_fpr = tpr[sorted_fpr_idx]

        return sorted_fpr, tpr

def true_pos(y_true, y_pred, dim=0):
    return torch.sum(y_true * torch.round(y_pred), dim=dim) # Only sum along H, W axis, assuming no C


def false_pos(y_true, y_pred, dim=0):
    return torch.sum(y_true * (1. - torch.round(y_pred)), dim=dim)


def false_neg(y_true, y_pred, dim=0):
    return torch.sum((1. - y_true) * torch.round(y_pred), dim=dim)


def precision(y_true, y_pred, dim):
    denom = (true_pos(y_true, y_pred, dim) + false_pos(y_true, y_pred, dim))
    denom = torch.clamp(denom, 10e-05)
    return true_pos(y_true, y_pred, dim) / denom

def recall(y_true, y_pred, dim):
    denom = (true_pos(y_true, y_pred, dim) + false_neg(y_true, y_pred, dim))
    denom = torch.clamp(denom, 10e-05)
    return true_pos(y_true, y_pred, dim) / denom

def f1_score(gts:torch.Tensor, preds:torch.Tensor, multi_threashold_mode=False, dim=(-1, -2)):
    # FIXME Does not operate proper
    gts = gts.float()
    preds = preds.float()

    if multi_threashold_mode:
        gts = gts[:, None, ...] # [B, Thresh, ...]
        gts = gts.expand_as(preds)

    with torch.no_grad():
        recall_val = recall(gts, preds, dim)
        precision_val = precision(gts, preds, dim)
        denom = torch.clamp( (recall_val + precision_val), 10e-5)

        f1 = 2. * recall_val * precision_val / denom

    return f1


def roc_score(y_true:torch.Tensor, y_preds:torch.Tensor, ):
    y_preds = y_preds.flatten().cpu().numpy()
    y_true = y_true.flatten().cpu().numpy()

    curve = roc_curve(y_true, y_preds, pos_label=1,  drop_intermediate=False)
    # print(curve)
    return curve
