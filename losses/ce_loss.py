import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    """Cross Entropy Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, stu_pred, label):
        loss = F.cross_entropy(stu_pred, label)
        return loss