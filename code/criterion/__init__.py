import torch.nn as nn

from .KD import DistillKL


def make_criterion_dict(kd_T):
    criterion_dict = nn.ModuleDict({})
    criterion_dict['cls'] = nn.CrossEntropyLoss()
    criterion_dict['kd'] = DistillKL(kd_T)
    return criterion_dict
