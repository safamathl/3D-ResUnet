"""

Dice loss
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # coefficient de Dice
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        return torch.clamp((1 - dice).mean(), 0, 1)
