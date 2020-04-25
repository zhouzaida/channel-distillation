import torch
import torch.nn as nn


class CDLoss(nn.Module):
    """Channel Distillation Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, stu_features: list, tea_features: list):
        loss = 0.
        for s, t in zip(stu_features, tea_features):
            s = s.mean(dim=(2, 3), keepdim=False)
            t = t.mean(dim=(2, 3), keepdim=False)
            loss += torch.mean(torch.pow(s - t, 2))
        return loss
