import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import math


class My(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        output = F.linear(F.normalize(input), F.normalize(self.weight))

        return output


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m_arc=0.50, m_am=0.0):
        super(ArcFace, self).__init__()

        self.s = s
        self.m_arc = m_arc
        self.m_am = m_am

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.cos_margin = math.cos(m_arc)
        self.sin_margin = math.sin(m_arc)
        self.min_cos_theta = math.cos(math.pi - m_arc)

    def forward(self, embbedings, label):
        embbedings = F.normalize(embbedings, dim=1)
        kernel_norm = F.normalize(self.weight, dim=0)
        # cos_theta = torch.mm(embbedings, kernel_norm).clamp(-1, 1)
        cos_theta = torch.mm(embbedings, kernel_norm).clip(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin

        # torch.where doesn't support fp16 input
        is_half = cos_theta.dtype == torch.float16

        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta.float() - self.m_am)
        if is_half:
            cos_theta_m = cos_theta_m.half()
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.s

        return output


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp

        return loss.mean()
