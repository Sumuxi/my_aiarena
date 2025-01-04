import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits):
        # 对学生和教师的 logits 进行温度缩放
        student_logits_temperature = student_logits / self.T
        teacher_logits_temperature = teacher_logits / self.T

        # 学生的对数概率分布（log_softmax 确保数值稳定性）
        student_probs_log = F.log_softmax(student_logits_temperature, dim=-1)
        # 教师的标准概率分布（softmax 输出标准概率）
        teacher_probs = F.softmax(teacher_logits_temperature, dim=-1)

        # 计算 KL 散度
        kl_div_loss = F.kl_div(student_probs_log, teacher_probs, reduction='batchmean')

        # 使用温度值作为权重调整 loss 值
        return kl_div_loss * (self.T ** 2)
