import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseDistillationLoss(nn.Module):
    """
    CWD (Channel-Wise KL) loss:
    For each (h, w): softmax over channel for teacher (C) and student (D),
    then average KL, then weighted sum across scales.
    """
    def __init__(self, scale_weights=(0.7, 0.2, 0.1), temperature=1.0, eps=1e-9):
        super().__init__()
        self.alpha = scale_weights
        self.T = temperature
        self.eps = eps

    def _cw_softmax(self, x):
        # x: (B, C, H, W) -> (B, C, H, W) softmax along C
        x = x / self.T
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x = x - x_max  # stability
        exp_x = torch.exp(x)
        sum_exp = torch.sum(exp_x, dim=1, keepdim=True) + self.eps
        return exp_x / sum_exp

    def _kl_div(self, p, q):
        # p, q: probs (B, C, H, W)
        kl = p * (torch.log(p + self.eps) - torch.log(q + self.eps))
        return torch.mean(torch.sum(kl, dim=1))  # average over B,H,W

    def forward(self, teacher_list, student_list):
        """
        teacher_list: [C3, C4, C5]
        student_list: [D3, D4, D5]
        """
        assert len(teacher_list) == len(student_list) == 3
        loss = 0.0
        for i, (C, D) in enumerate(zip(teacher_list, student_list)):
            p = self._cw_softmax(C)
            q = self._cw_softmax(D)
            loss += self.alpha[i] * self._kl_div(p, q)
        return loss * (self.T * self.T)