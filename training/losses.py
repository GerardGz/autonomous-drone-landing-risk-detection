import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Computes Dice Loss for binary segmentation
    Dice = 2 * (pred ∩ target) / (pred + target)
    """
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, target):
        """
        Args:
            logits: output from model before sigmoid, shape (B, 1, H, W)
            target: ground truth mask, shape (B, 1, H, W) or (B, H, W)
        """
        if target.dim() == 3:  # add channel dim if missing
            target = target.unsqueeze(1)

        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice
        return loss.mean()


class BCEWithLogitsDiceLoss(nn.Module):
    """
    Combined BCEWithLogits + Dice Loss
    """
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super(BCEWithLogitsDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.w_bce = weight_bce
        self.w_dice = weight_dice

    def forward(self, logits, target):
        loss_bce = self.bce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.w_bce * loss_bce + self.w_dice * loss_dice



# Optional helper function

def get_loss(logits, target, combined=True):
    if combined:
        criterion = BCEWithLogitsDiceLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    return criterion(logits, target)
