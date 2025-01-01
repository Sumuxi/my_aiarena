import logging
import os
import time

import tensorboard_logger as tb_logger
import torch


# TODO：loss 和 acc，分英雄、分动作、以及细致的15个loss 15个 acc
# TODO：拆分loss和validate，若干steps或者epochs后进行一次验证，定义多少个step为一个epoch？ validate需要使用验证集
# TODO: 记录每一层的梯度，记录每一层的参数，方便查看梯度变化、参数分布变化，方便可视化看结果
# TODO: 完善验证集dataset的输入和处理
# TODO: 每个模型文件，写不同计算量的模型
# TODO: loss函数，目前是2中，分类 loss + KD loss，引入更多种类的loss?
# TODO：学习率衰减、batch_size衰减
# TODO: optimizer，定义不同的优化器，SGD、AdamW
# TODO: 多机多卡训练？

# 定义训练器
class Trainer:
    def __init__(self,
                 model,
                 criterion,
                 train_dataset,
                 test_dataset,
                 config
                 ):
        self.model = model
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tb_logger = tb_logger.Logger(logdir=self.config.log_dir, flush_secs=2)
        self.start_time = None
        self._init_env()
        self._init_model()

    def _init_optimizer(self, optim):
        config = self.config
        parameters = self.model.parameters()
        if optim == 'adam':
            optimizer = torch.optim.Adam(
                params=parameters, lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8
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

    def _init_env(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.local_rank)
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"Use {self.device} as default device")

    def _init_model(self):
        self.local_step = 0
        self.optimizer = self._init_optimizer(self.config.optimizer)

        self.parameters = [
            p
            for param_group in self.optimizer.param_groups
            for p in param_group["params"]
        ]
        # load init model
        if self.config.use_init_model:
            # init_model_path直接指定模型文件名
            init_model_path = self.config.init_model_path  # 绝对路径
            self._load_checkpoint(init_model_path)

        get_lr_scheduler = getattr(self.model, "get_lr_scheduler", None)
        if callable(get_lr_scheduler):
            self.lr_scheduler = self.model.get_lr_scheduler(
                self.optimizer, self.local_step
            )
        else:
            if self.config.use_lr_decay:
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.T_max,
                    eta_min=self.config.lr_end)
            else:
                self.lr_scheduler = None

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.local_step = torch.tensor(self.local_step, dtype=torch.int).to(self.device)

    def _load_checkpoint(self, init_model_path):
        state_dict = torch.load(init_model_path, map_location="cpu")
        if "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict["network_state_dict"], strict=True
        )
        self.logger.info(
            f"load ckpt success, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
        )
        ckpt_step = state_dict.get("step", 0)
        self.logger.info(f"loaded checkpoint from {init_model_path}, step: {ckpt_step}")
        self.local_step = 0 if ckpt_step is None else ckpt_step

    def _save_checkpoint(self, model, optimizer, checkpoint_dir: str, step: int):
        os.makedirs(checkpoint_dir, exist_ok=True)
        step = int(step)
        self.last_save_step = step
        checkpoint_file = os.path.join(checkpoint_dir, f"model_step{step}.pth")
        torch.save(
            {
                "network_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            },
            checkpoint_file,
        )

    def run(self):
        self.start_time = time.time()
        self.model.train()
        while self.local_step < self.config.max_steps:
            x, y_t, z_t = self.train_dataset.get_next_batch()
            # x: state
            # y_t: teacher action (label)
            # z_t: teacher logits
            if torch.cuda.is_available():
                x = x.to(self.device, non_blocking=True)
                y_t = y_t.to(self.device, non_blocking=True)
                z_t = z_t.to(self.device, non_blocking=True)

            # increment local step
            self.local_step += 1
            # 1.forward
            # 2.backward
            # 3.get loss and some metrics
            result_dict = self._run_step(x, y_t, z_t)
            self._after_run_step(result_dict)

    def _run_step(self, x, y_t, z_t):
        """
        Perform a single training step.

        Args:
            x: Input data.
            y_t: Target labels for supervised tasks.
            z_t: Target output for distillation tasks or other auxiliary tasks.

        Returns:
            result_dict: A dictionary containing loss, accuracy, gradient norms, and other metrics.
        """
        # Forward pass
        z_s = self.model(x)
        result_dict = self._loss(z_s, z_t, y_t)
        loss = result_dict['loss']

        # Zero gradients
        self.optimizer.zero_grad()
        # Backward pass
        loss.backward()

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


    # def _compute_loss_and_accuracy(self, z_s, z_t, y_t):
    #     criterion_cls = self.criterion['cls']
    #     criterion_div = self.criterion['kd']
    #
    #     # total_dims = sum([13, 25, 42, 42, 39])
    #     # z_s = z_s.reshape(3, -1, total_dims)
    #     # z_t = z_t.reshape(3, -1, total_dims)
    #     # y_t = y_t.reshape(3, -1, 5)
    #     # bs = z_s.shape[0]
    #     # student_logits_list = z_s.split([13, 25, 42, 42, 39], dim=-1)  # (bs, action_dim)
    #     # teacher_logits_list = z_t.split([13, 25, 42, 42, 39], dim=-1)  # (bs, action_dim)
    #     # teacher_action_list = y_t.split([1] * 5, dim=-1)  # (bs, 1)
    #
    #     action_dims = [13, 25, 42, 42, 39]
    #     student_logits_list = [z_s[hero_index].split(action_dims, dim=-1)
    #                            for hero_index in range(3)]  # 3 heroes, 5 actions, tensor.shape = (bs, action_dim)
    #     teacher_logits_list = [z_t[hero_index].split(action_dims, dim=-1)
    #                            for hero_index in range(3)]
    #     teacher_action_list = [y_t[hero_index].split([1] * 5, dim=-1)
    #                            for hero_index in range(3)]
    #
    #     loss_list = torch.zeros((3, 5, 2))  # 3 heroes, 5 actions, 2 loss [hard loss, kd loss]
    #     # hard loss, classification loss
    #     # kd loss, KL divergence loss
    #     for i in range(3):
    #         for j in range(5):
    #             cls_loss = criterion_cls(student_logits_list[i][j], teacher_action_list[i][j].squeeze(-1).long())
    #             div_loss = criterion_div(student_logits_list[i][j], teacher_logits_list[i][j])
    #             loss_list[i][j][0] = cls_loss
    #             loss_list[i][j][1] = div_loss
    #     hard_loss = torch.sum(loss_list[:, :, 0])
    #     kd_loss = torch.sum(loss_list[:, :, 1])
    #     loss_list = loss_list.sum(dim=0)
    #     result_dict = {
    #         'loss': kd_loss,
    #         'hard_loss': hard_loss,
    #         'kd_loss': kd_loss,
    #         'loss_list': loss_list,
    #     }
    #
    #     # accuracy
    #     if self.local_step % self.config.print_freq == 0 \
    #             or self.local_step % self.config.tb_log_freq == 0:
    #         acc_list = torch.zeros((3, 5, 2))  # 3 heroes, 5 actions, 2 acc [top1_acc, top5_acc]
    #         for i in range(3):
    #             for j in range(5):
    #                 top1_acc, top5_acc = self._top_k_accuracy(student_logits_list[i][j], teacher_action_list[i][j],
    #                                                           topk=(1, 5))
    #                 acc_list[i][j] = torch.Tensor([top1_acc, top5_acc])
    #         acc_list = acc_list.mean(dim=0)  # 5 actions, 2 acc [top1_acc, top5_acc]
    #         result_dict['acc_list'] = acc_list
    #     return result_dict

    def _loss(self, z_s, z_t, y_t):
        criterion_cls = self.criterion['cls']
        criterion_div = self.criterion['kd']

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
            'hard_loss': hard_loss,
            'kd_loss': kd_loss,
            'loss_list': loss_list,
        }
        return result_dict

    def _validate(self, z_s, z_t, y_t):
        action_dims = [13, 25, 42, 42, 39]
        student_logits_list = [z_s[hero_index].split(action_dims, dim=-1)
                               for hero_index in range(3)]  # 3 heroes, 5 actions, tensor.shape = (bs, action_dim)
        teacher_logits_list = [z_t[hero_index].split(action_dims, dim=-1)
                               for hero_index in range(3)]
        teacher_action_list = [y_t[hero_index].split([1] * 5, dim=-1)
                               for hero_index in range(3)]

        acc_list = torch.zeros((3, 5, 2))  # 3 heroes, 5 actions, 2 acc [top1_acc, top5_acc]
        for i in range(3):
            for j in range(5):
                top1_acc, top5_acc = self._top_k_accuracy(student_logits_list[i][j], teacher_action_list[i][j],
                                                          topk=(1, 5))
                acc_list[i][j] = torch.Tensor([top1_acc, top5_acc])
        return acc_list

    def _top_k_accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            if batch_size == 0:
                return [torch.tensor(0.0) for _ in topk]

            # 获取 top-k 索引
            top_k_preds = output.topk(maxk, dim=1, largest=True, sorted=True).indices

            # 检查预测是否正确
            correct = top_k_preds.eq(target.view(-1, 1).expand_as(top_k_preds))

            # 计算每个 k 的准确率
            res = []
            for k in topk:
                correct_k = correct[:, :k].sum()  # 正确的样本数
                res.append(correct_k / batch_size)
            return res

    def _after_run_step(self, result_dict):
        # save model
        if self.local_step % self.config.save_model_freq == 0:
            self._save_checkpoint(
                self.model,
                self.optimizer,
                self.config.save_model_dir,
                self.local_step,
            )

        if self.local_step % self.config.print_freq == 0 \
                or self.local_step % self.config.tb_log_freq == 0:
            total_step = int(self.local_step)
            hard_loss = result_dict['hard_loss'].detach().cpu().item()
            kd_loss = result_dict['kd_loss'].detach().cpu().item()
            # print log
            if self.local_step % self.config.print_freq == 0:
                self.logger.info('Step {step}\t'
                                 'Hard Loss {hard_loss:.4f}\t'
                                 'KD Loss {kd_loss:.4f}\t'
                                 .format(step=total_step,
                                         hard_loss=hard_loss,
                                         kd_loss=kd_loss))

            # write tensorboard log
            if self.local_step % self.config.tb_log_freq == 0:
                elapsed_time = time.time() - self.start_time
                self.start_time = time.time()
                generation_rate, consumption_rate = self.train_dataset.get_rate()

                # training metrics info
                self.tb_logger.log_value("metrics/avg_step_time", elapsed_time / self.config.tb_log_freq, total_step)
                self.tb_logger.log_value("metrics/G rate", generation_rate, total_step)
                self.tb_logger.log_value("metrics/C rate", consumption_rate, total_step)

                # loss
                self.tb_logger.log_value("loss/cls", hard_loss, total_step)
                self.tb_logger.log_value("loss/kd", kd_loss, total_step)

                # 3 heroes, 5 actions, 2 loss [hard loss, kd loss]
                loss_list = result_dict['loss_list'].detach().cpu()
                hero_loss_list = loss_list.sum(dim=1).numpy().tolist()
                action_loss_list = loss_list.sum(dim=0).numpy().tolist()
                loss_list = loss_list.numpy().tolist()
                for i in range(3):
                    self.tb_logger.log_value(f"hard_loss/heroes/hero{i + 1}", hero_loss_list[i][0], total_step)
                    self.tb_logger.log_value(f"kd_loss/heroes/hero{i + 1}", hero_loss_list[i][1], total_step)
                for i in range(5):
                    self.tb_logger.log_value(f"hard_loss/actions/act{i + 1}", action_loss_list[i][0], total_step)
                    self.tb_logger.log_value(f"kd_loss/actions/act{i + 1}", action_loss_list[i][1], total_step)
                for i in range(3):
                    for j in range(5):
                        self.tb_logger.log_value(f"hard_loss/act{j + 1}", loss_list[i][j][0], total_step)
                        self.tb_logger.log_value(f"kd_loss/act{j + 1}", loss_list[i][j][1], total_step)

                # 3 heroes, 5 actions, 2 acc [top1_acc, top5_acc]
                # acc_list = result_dict['acc_list'].detach().cpu().numpy().tolist()
                # for idx, acc in enumerate(acc_list):
                #     self.tb_logger.log_value(f"top1_acc/act{idx + 1}", acc[0], total_step)
                #     self.tb_logger.log_value(f"top5_acc/act{idx + 1}", acc[1], total_step)

    def _after_run_epoch(self):
        pass
