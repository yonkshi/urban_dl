import torch
from sklearn.metrics import roc_auc_score, roc_curve


class MultiThresholdMetric():
    def __init__(self, y_true:torch.Tensor, y_pred, threshold):

        # FIXME Does not operate properly

        '''
        Takes in rasterized and batched images
        :param y_true: [B, H, W]
        :param y_pred: [B, C, H, W]
        :param threshold: [Thresh]
        '''
        self._y_true = y_true.bool()
        self._y_pred = y_pred.squeeze()
        B, C, H, W = y_pred.shape
        self.numel = C * H * W # total number of pixels per image
        self._thresholds = threshold
        self._data_dims = (-1, -2, -3, -4) # For a B/W image, it should be [Thresh, B, C, H, W],

        # self._normalize_dimensions()
        # self._build_threshold_for_computation()
        self._pre_compute_basic_metrics()

    def _pre_compute_basic_metrics(self):
        shape = self._thresholds.shape
        self.TP = torch.empty(*shape)
        self.TN = torch.empty(*shape)
        self.FP = torch.empty(*shape)
        self.FN = torch.empty(*shape)
        # Running it sequentially because vectorized form is too big to be fit inside the memory.
        print('precomputing basic metrics..')
        for i, threshold in enumerate(self._thresholds):
            y_pred_offset = (self._y_pred - threshold + 0.5).round().bool()

            self.TP[i] = (self._y_true & y_pred_offset).sum()
            self.TN[i] = (~self._y_true & ~y_pred_offset).sum()
            self.FP[i] = (self._y_true & ~y_pred_offset).sum()
            self.FN[i] = (~self._y_true & y_pred_offset).sum()
            print(i, end=' ')
            # self.TP = (self._y_true * torch.round(self._y_pred)).sum(dim=self._data_dims)
        print('completed')

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

    def compute_basic_metrics(self):
        '''
        Computes False Negative Rate and False Positive rate
        :return:
        '''

        false_pos_rate = self.FP/(self.FP + self.TN)
        false_neg_rate = self.FN / (self.FN + self.TP)

        return false_pos_rate, false_neg_rate

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
