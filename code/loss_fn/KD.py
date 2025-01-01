import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    # def forward(self, y_s, y_t):
    #     p_s = F.log_softmax(y_s / self.T, dim=1)
    #     p_t = F.softmax(y_t / self.T, dim=1)
    #     loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
    #     return loss

    def forward(self, student_logits, teacher_logits):
        student_logits_temperature = student_logits / self.T
        teacher_logits_temperature = teacher_logits / self.T

        # 使用 log_softmax, 避免指数运行带来的数值不稳定
        student_probs_log = F.log_softmax(student_logits_temperature, dim=-1)
        teacher_probs_log = F.log_softmax(teacher_logits_temperature, dim=-1)

        # 计算 KL 散度，log_target=True 表示教师模型的目标已是 log 概率
        kl_div_loss = F.kl_div(student_probs_log, teacher_probs_log, reduction='none', log_target=True)
        # 对类别维度求和，得到批量中每个样本的损失
        kl_div_loss = kl_div_loss.sum(dim=-1) * (self.T ** 2)
        # 对批量维度求均值
        return kl_div_loss.mean()
