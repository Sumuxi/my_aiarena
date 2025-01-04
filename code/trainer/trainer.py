import logging
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from criterion import make_criterion_dict
from datasets.hok_3v3 import FrameX1
from helper.profile_util import torch_profile
from models import make_model


def save_checkpoint(checkpoint_dir: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler=None,
                    step: int = 0,
                    rng_state=None,
                    cuda_rng_state=None):
    """
    保存训练检查点，包括模型状态、优化器状态、调度器状态、训练步数和随机数状态。

    Args:
        checkpoint_dir (str): 检查点保存目录。
        model (torch.nn.Module): 待保存的模型。
        optimizer (torch.optim.Optimizer): 待保存的优化器。
        lr_scheduler (torch.optim.lr_scheduler, optional): 学习率调度器，默认为 None。
        step (int): 当前训练步数。
        rng_state: CPU 随机数状态，默认为 None。
        cuda_rng_state: GPU 随机数状态，默认为 None。
    """
    # 检查输入参数
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Expected 'model' to be an instance of torch.nn.Module.")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("Expected 'optimizer' to be an instance of torch.optim.Optimizer.")

    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 检查点文件路径
    checkpoint_file = os.path.join(checkpoint_dir, f"model_step{step}.pth")

    # 准备保存的状态字典
    checkpoint = {
        "network_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }

    # 如果有学习率调度器，保存其状态
    if lr_scheduler is not None:
        checkpoint["scheduler_state_dict"] = lr_scheduler.state_dict()

    # 保存随机数状态（如果提供）
    if rng_state is not None:
        checkpoint["rng_state"] = rng_state
    if cuda_rng_state is not None and torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = cuda_rng_state

    # 保存检查点
    torch.save(checkpoint, checkpoint_file)


def load_checkpoint(checkpoint_path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    lr_scheduler=None):
    """
    从检查点文件加载模型、优化器、调度器及训练状态。

    Args:
        checkpoint_path (str): 检查点文件路径。
        model (torch.nn.Module): 待加载的模型。
        optimizer (torch.optim.Optimizer, optional): 待加载的优化器，默认为 None。
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 待加载的学习率调度器，默认为 None。

    Returns:
        step (int): 恢复的训练步数。
    """
    # 检查文件路径是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist.")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 恢复模型状态
    if "network_state_dict" not in checkpoint:
        raise KeyError("The checkpoint does not contain 'network_state_dict'.")
    model.load_state_dict(checkpoint["network_state_dict"])

    # 恢复优化器状态（如果提供）
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 恢复学习率调度器状态（如果提供）
    if lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 恢复训练进度和随机数状态
    step = checkpoint.get("step", 0)
    rng_state = checkpoint.get("rng_state", None)
    cuda_rng_state = checkpoint.get("cuda_rng_state", None)

    if rng_state is not None:
        torch.set_rng_state(rng_state)

    if cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)

    return step


def loss_fn(criterion, z_s, z_t, y_t):
    criterion_cls = criterion['cls']
    criterion_div = criterion['kd']

    action_dims = [13, 25, 42, 42, 39]
    student_logits_list = [z_s[hero_index].split(action_dims, dim=-1)
                           for hero_index in range(3)]  # 3 heroes, 5 actions, tensor.shape = (bs, action_dim)
    teacher_logits_list = [z_t[hero_index].split(action_dims, dim=-1)
                           for hero_index in range(3)]
    teacher_action_list = [y_t[hero_index].split([1] * 5, dim=-1)
                           for hero_index in range(3)]

    # 可以分3个英雄看不同英雄的loss
    # 可以分5个动作来看不同动作的loss
    # 总：可以看15个loss
    # 3 heroes, 5 actions, 2 loss [hard loss, kd loss]
    loss_list = torch.zeros((3, 5, 2))

    # hard loss, classification loss
    # kd loss, KL divergence loss
    for i in range(3):
        for j in range(5):
            cls_loss = criterion_cls(student_logits_list[i][j], teacher_action_list[i][j].squeeze(-1).long())
            div_loss = criterion_div(student_logits_list[i][j], teacher_logits_list[i][j])
            loss_list[i][j][0] = cls_loss
            loss_list[i][j][1] = div_loss
    hard_loss = torch.sum(loss_list[:, :, 0])
    kd_loss = torch.sum(loss_list[:, :, 1])
    # loss_list = loss_list.sum(dim=0)
    result_dict = {
        'loss': kd_loss,
        'hard_loss': hard_loss.detach().cpu(),
        'kd_loss': kd_loss.detach().cpu(),
        'loss_list': loss_list.detach().cpu(),
    }
    return result_dict


def top_k_accuracy(output, target, mask, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        # 获取 top-k 索引
        top_k_preds = output.topk(maxk, dim=1, largest=True, sorted=True).indices

        # 检查预测是否正确
        correct = top_k_preds.eq(target.view(-1, 1).expand_as(top_k_preds)) & mask.int()

        # 计算每个 k 的准确率
        result = []
        for k in topk:
            correct_k = correct[:, :k].sum()  # 正确的样本数
            result.append(correct_k / torch.sum(mask))  # batch_size 改成 torch.sum(mask)
        return result


def accuracy_fn(z_s, z_t, y_t, m_t):
    action_dims = [13, 25, 42, 42, 39]
    student_logits_list = [z_s[hero_index].split(action_dims, dim=-1)
                           for hero_index in range(3)]  # 3 heroes, 5 actions, tensor.shape = (bs, action_dim)
    # teacher_logits_list = [z_t[hero_index].split(action_dims, dim=-1)
    #                        for hero_index in range(3)]
    teacher_action_list = [y_t[hero_index].split([1] * 5, dim=-1)
                           for hero_index in range(3)]
    sub_action_mask_list = [m_t[hero_index].split([1] * 5, dim=-1)
                            for hero_index in range(3)]

    acc_list = torch.zeros((3, 5, 2))  # 3 heroes, 5 actions, 2 acc [top1_acc, top5_acc]
    for i in range(3):
        for j in range(5):
            top1_acc, top5_acc = top_k_accuracy(student_logits_list[i][j],
                                                teacher_action_list[i][j],
                                                sub_action_mask_list[i][j],
                                                topk=(1, 5))
            acc_list[i][j] = torch.Tensor([top1_acc, top5_acc])
    return acc_list.detach().cpu()


def evaluate(config, train_step):
    test_dataset = FrameX1(root_dir=config.valid_data_dir,
                           batch_size=256 * 16,
                           batch_queue_size=50,
                           num_reader=2
                           )
    # max_steps = 20 * 1000 * 16 // (32 * 16)
    max_steps = 1
    validate_step = 0

    model = make_model(config.model)
    checkpoint_file = os.path.join(config.save_model_dir, f"model_step{train_step}.pth")
    load_checkpoint(checkpoint_file, model)

    criterion = make_criterion_dict(config.kd_T)

    if torch.cuda.is_available():
        device = torch.device("cuda", config.local_rank)
    else:
        device = torch.device("cpu")

    model = model.to(device)
    criterion = criterion.to(device)
    with torch.no_grad():
        result_dict = {
            'hard_loss': [],
            'kd_loss': [],
            'loss_list': [],
            'acc_list': [],
        }
        while validate_step < max_steps:
            x, y_t, z_t, m_t = test_dataset.get_next_batch()
            # x: state
            # y_t: teacher action (label)
            # z_t: teacher logits
            # m_t: sub_action_mask
            if torch.cuda.is_available():
                x = x.to(device, non_blocking=True)
                y_t = y_t.to(device, non_blocking=True)
                z_t = z_t.to(device, non_blocking=True)
                m_t = m_t.to(device, non_blocking=True)

            # increment local step
            validate_step += 1
            # forward
            z_s = model(x)
            loss_dict = loss_fn(criterion, z_s, z_t, y_t)
            acc_list = accuracy_fn(z_s, z_t, y_t, m_t)
            result_dict['hard_loss'].append(loss_dict['hard_loss'])
            result_dict['kd_loss'].append(loss_dict['kd_loss'])
            result_dict['loss_list'].append(loss_dict['loss_list'])
            result_dict['acc_list'].append(acc_list)

        result_dict['hard_loss'] = torch.stack(result_dict['hard_loss']).mean(dim=0)
        result_dict['kd_loss'] = torch.stack(result_dict['kd_loss']).mean(dim=0)
        result_dict['loss_list'] = torch.stack(result_dict['loss_list']).mean(dim=0)
        result_dict['acc_list'] = torch.stack(result_dict['acc_list']).mean(dim=0)

    result_dict['step'] = train_step
    print(f"evaluate step {train_step} done.", flush=True)
    return result_dict


# loss 和 acc，分英雄、分动作、以及细致的15个loss 15个 acc
# 记录每一层的梯度，记录每一层的参数，方便查看梯度变化、参数分布变化，方便可视化看结果
# 检查 dataset 的样本处理维度
# 构造验证集
# 在验证集进行validate, 若干steps或者epochs后进行一次验证，定义多少个step为一个epoch？
# 每个模型文件，写不同计算量的模型
# loss函数，目前是2中，分类 loss + KD loss，引入更多种类的loss?
# optimizer，定义不同的优化器，SGD、AdamW
# TODO：学习率衰减、batch_size衰减
# TODO: 多机多卡训练？


class Trainer:
    """
    定义训练器
    """

    def __init__(self, config):
        self.config = config
        self.model = make_model(config.model)
        self.criterion = make_criterion_dict(config.kd_T)

        self.train_dataset = FrameX1(root_dir=config.train_data_dir,
                                     batch_size=config.batch_size * 16,  # 为了和之前的连续16帧时的数据集上的batch_size对齐
                                     batch_queue_size=config.batch_queue_size,
                                     num_reader=config.num_reader
                                     )

        self.logger = logging.getLogger(__name__)
        self.tb_logger = SummaryWriter(log_dir=self.config.log_dir, flush_secs=60)

        self.start_time = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.local_rank)
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"Use {self.device} as default device")

        self._init_model()
        self._profile()

    def _init_optimizer(self, optim):
        config = self.config
        parameters = self.model.parameters()
        if optim == 'adam':
            optimizer = torch.optim.Adam(
                params=parameters, lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8
            )
        elif optim == 'adamw':
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=config.weight_decay
            )
        elif optim == 'sgd':
            optimizer = torch.optim.SGD(params=parameters,
                                        lr=config.learning_rate,
                                        momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        else:
            optimizer = torch.optim.SGD(params=parameters,
                                        lr=config.learning_rate,
                                        momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        return optimizer

    def _init_model(self):
        self.local_step = 0
        self.optimizer = self._init_optimizer(self.config.optimizer)

        self.parameters = [
            p
            for param_group in self.optimizer.param_groups
            for p in param_group["params"]
        ]

        if self.config.use_lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.T_max,
                eta_min=self.config.lr_end)
        else:
            self.lr_scheduler = None

        # resume a saved model
        if self.config.use_init_model:
            # init_model_path直接指定模型文件名
            init_model_path = self.config.init_model_path  # 绝对路径
            step = load_checkpoint(init_model_path,
                                   self.model,
                                   self.optimizer,
                                   self.lr_scheduler)
            self.local_step = step

    def _profile(self):
        x = torch.rand(3, 1, 4586)
        flops, params = torch_profile(self.model, x)
        self.logger.info(f"model: {self.config.model}, "
                         f"{flops / (1000 ** 2) / 673.437096 * 100: .2f}%, "
                         f"FLOPs = {flops / (1000 ** 2)}M, "
                         f"Params = {params / (1000 ** 2)}M.")

    def run(self):
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.start_time = time.time()
        self.model.train()
        while self.local_step < self.config.max_steps:
            x, y_t, z_t, m_t = self.train_dataset.get_next_batch()
            # x: state
            # y_t: teacher action (label)
            # z_t: teacher logits
            # m_t: sub_action_mask
            if torch.cuda.is_available():
                x = x.to(self.device, non_blocking=True)
                y_t = y_t.to(self.device, non_blocking=True)
                z_t = z_t.to(self.device, non_blocking=True)
                m_t = m_t.to(self.device, non_blocking=True)

            # increment local step
            self.local_step += 1
            # 1.forward
            # 2.backward
            # 3.get loss and some metrics
            result_dict = self._run_step(x, y_t, z_t)
            if self.local_step % self.config.log_freq == 0:
                result_dict['acc_list'] = accuracy_fn(result_dict['z_s'], z_t, y_t, m_t)
            self._after_run_step(result_dict)

    def _run_step(self, x, y_t, z_t):
        """
        Perform a single training step.

        Args:
            x: input data.
            y_t: teacher action (labels) for supervised tasks.
            z_t: teacher logits for distillation tasks or other auxiliary tasks.

        Returns:
            result_dict: A dictionary containing loss, accuracy, gradient norms, and other metrics.
        """
        # Forward pass
        z_s = self.model(x)
        result_dict = loss_fn(self.criterion, z_s, z_t, y_t)
        loss = result_dict['loss']
        result_dict['z_s'] = z_s

        # Zero gradients
        self.optimizer.zero_grad()
        # Backward pass
        loss.backward()

        # if self.local_step % self.config.log_freq == 0:
        #     # 记录每一层的参数、梯度及梯度范数
        #     params = {}
        #     grads = {}
        #     grad_norms = {}
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             params[name] = param.detach().cpu()
        #             grads[name] = param.grad.detach().cpu()
        #             grad_norms[name] = param.grad.norm().cpu().item()
        #     result_dict['params'] = params
        #     result_dict['grads'] = grads
        #     result_dict['grad_norms'] = grad_norms
        # result_dict['acc_list'] = accuracy_fn(z_s, z_t, y_t, m_t)

        # Gradient norm (pre-clipping)
        # l2_norm_total = torch.sqrt(sum(torch.norm(p.grad, 2).item()**2 for p in model.parameters() if p.grad is not None))
        pre_clip_total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None]),
            2
        )
        result_dict["pre_clip_total_norm"] = pre_clip_total_norm.item()
        # Clip gradient norm if enabled
        if self.config.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_range
            )

            # Gradient norm (post-clipping)
            post_clip_total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None]),
                2
            )
            result_dict["post_clip_total_norm"] = post_clip_total_norm.item()

        # Update parameters
        self.optimizer.step()
        return result_dict

    # def _write_eval_log(self, future):
    #     try:
    #         result_dict = future.result()  # 获取任务的结果
    #
    #         total_step = result_dict['step']
    #
    #         # loss
    #         hard_loss = result_dict['hard_loss'].item()
    #         kd_loss = result_dict['kd_loss'].item()
    #         # 3 heroes, 5 actions, 2 loss [hard loss, kd loss]
    #         loss_list = result_dict['loss_list']
    #         hero_loss_list = loss_list.sum(dim=1).numpy().tolist()
    #         action_loss_list = loss_list.sum(dim=0).numpy().tolist()
    #         loss_list = loss_list.numpy().tolist()
    #
    #         # accuracy
    #         # 3 heroes, 5 actions, 2 acc [top1_acc, top5_acc]
    #         acc_list = result_dict['acc_list']
    #         hero_acc_list = acc_list.mean(dim=1).numpy().tolist()
    #         action_acc_list = acc_list.mean(dim=0).numpy().tolist()
    #         acc_list = acc_list.numpy().tolist()
    #
    #         eval_loss_acc = {
    #             'step': total_step,
    #
    #             'hard_loss': hard_loss,
    #             'kd_loss': kd_loss,
    #             'hero_loss_list': hero_loss_list,
    #             'action_loss_list': action_loss_list,
    #             'loss_list': loss_list,
    #
    #             'hero_acc_list': hero_acc_list,
    #             'action_acc_list': action_acc_list,
    #             'acc_list': acc_list,
    #         }
    #         self.eval_logger.info(eval_loss_acc)
    #     except Exception as e:
    #         self.logger.error(f"evaluation task raised an exception: {e}")

    def _after_run_step(self, result_dict):
        # save model
        if self.local_step % self.config.save_model_freq == 0:
            save_checkpoint(
                self.config.save_model_dir,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.local_step,
                torch.get_rng_state(),
                torch.cuda.get_rng_state(),
            )
            self.logger.info(f"saved checkpoint, step = {self.local_step}.")

        if self.local_step % self.config.log_freq == 0:
            total_step = int(self.local_step)

            # loss
            hard_loss = result_dict['hard_loss'].item()
            kd_loss = result_dict['kd_loss'].item()
            # 3 heroes, 5 actions, 2 loss [hard loss, kd loss]
            loss_list = result_dict['loss_list']
            hero_loss_list = loss_list.sum(dim=1).numpy().tolist()
            action_loss_list = loss_list.sum(dim=0).numpy().tolist()

            # accuracy
            # 3 heroes, 5 actions, 2 acc [top1_acc, top5_acc]
            acc_list = result_dict['acc_list']
            hero_acc_list = acc_list.mean(dim=1).numpy().tolist()
            action_acc_list = acc_list.mean(dim=0).numpy().tolist()

            np.savez_compressed(os.path.join(self.config.save_npz_log_dir, f"train-loss-acc_{total_step}.npz"),
                                step=total_step,
                                loss=loss_list.numpy(),
                                acc=acc_list.numpy())
            loss_list = loss_list.numpy().tolist()
            # acc_list = acc_list.numpy().tolist()

            # metrics info
            elapsed_time = time.time() - self.start_time
            self.start_time = time.time()
            avg_step_time = elapsed_time / self.config.log_freq
            generation_rate, consumption_rate = self.train_dataset.get_rate()
            pre_clip_total_norm = result_dict["pre_clip_total_norm"]
            post_clip_total_norm = result_dict["post_clip_total_norm"]

            # print log
            self.logger.info(('Step {step}\t'
                              'Hard Loss {hard_loss:.4f}\t'
                              'KD Loss {kd_loss:.4f}\t'
                              'Act1 Top1 Acc {act1_top1_acc:.4f}\t'
                              'G Rate {generation_rate:.4f}\t'
                              'C Rate {consumption_rate:.4f}\t'
                              ).format(
                step=total_step,
                hard_loss=hard_loss,
                kd_loss=kd_loss,
                act1_top1_acc=action_acc_list[0][0],
                generation_rate=generation_rate,
                consumption_rate=consumption_rate
            ))

            # write tensorboard log
            # training metrics info
            self.tb_logger.add_scalar("metrics/avg_step_time", avg_step_time, total_step)
            self.tb_logger.add_scalar("metrics/generation_rate", generation_rate, total_step)
            self.tb_logger.add_scalar("metrics/consumption_rate", consumption_rate, total_step)

            # gradient norm
            self.tb_logger.add_scalar("train/pre_clip_total_norm", pre_clip_total_norm, total_step)
            self.tb_logger.add_scalar("train/post_clip_total_norm", post_clip_total_norm, total_step)

            # loss
            self.tb_logger.add_scalar("loss/cls", hard_loss, total_step)
            self.tb_logger.add_scalar("loss/kd", kd_loss, total_step)

            # 3 heroes loss
            for i in range(3):
                self.tb_logger.add_scalar(f"hard_loss/heroes/hero{i + 1}", hero_loss_list[i][0], total_step)
                self.tb_logger.add_scalar(f"kd_loss/heroes/hero{i + 1}", hero_loss_list[i][1], total_step)
            # # 5 actions loss
            for i in range(5):
                self.tb_logger.add_scalar(f"hard_loss/actions/act{i + 1}", action_loss_list[i][0], total_step)
                self.tb_logger.add_scalar(f"kd_loss/actions/act{i + 1}", action_loss_list[i][1], total_step)
            # loss_list, 3 x 5
            for i in range(3):
                for j in range(5):
                    self.tb_logger.add_scalar(f"hard_loss/hero{i + 1}_act{j + 1}", loss_list[i][j][0], total_step)
                    self.tb_logger.add_scalar(f"kd_loss/hero{i + 1}_act{j + 1}", loss_list[i][j][1], total_step)

            # 3 heroes acc
            # for i in range(3):
            #     self.tb_logger.add_scalar(f"top1_acc/heroes/hero{i + 1}", hero_acc_list[i][0], total_step)
            #     self.tb_logger.add_scalar(f"top5_acc/heroes/hero{i + 1}", hero_acc_list[i][1], total_step)
            # 5 actions acc
            for i in range(5):
                self.tb_logger.add_scalar(f"top1_acc/actions/act{i + 1}", action_acc_list[i][0], total_step)
                self.tb_logger.add_scalar(f"top5_acc/actions/act{i + 1}", action_acc_list[i][1], total_step)
            # acc_list, 3 x 5
            # for i in range(3):
            #     for j in range(5):
            #         self.tb_logger.add_scalar(f"hard_loss/hero{i + 1}_act{j + 1}", acc_list[i][j][0], total_step)
            #         self.tb_logger.add_scalar(f"kd_loss/hero{i + 1}_act{j + 1}", acc_list[i][j][1], total_step)

    # def _write_json_log(self, file_name, log_dict):
    #     file_path = os.path.join(self.config.log_dir, file_name)
    #     with open(os.path.join(self.config.log_dir, file_name), "a") as file:
    #         file.write("This is an additional line.\n")
    #
    #     # 读取现有 JSON 文件内容
    #     with open(file_path, "r") as f:
    #         data = json.load(f)
    #         if not isinstance(data, list):
    #             raise ValueError("JSON 文件必须是一个数组结构")
    #
    #     # 追加新数据
    #     data.append(log_dict)
    #
    #     # 将更新后的数据写回文件
    #     with open(file_path, "w") as f:
    #         json.dump(data, f, indent=4, ensure_ascii=False)
