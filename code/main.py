import argparse
import logging
import os
import random

import numpy as np
import torch

from trainer.trainer import Trainer


def set_seed(seed: int = 56, deterministic: bool = True):
    """
    设置随机种子以确保结果可复现。

    Args:
        seed (int): 随机种子值。
        deterministic (bool): 是否启用确定性算法，默认为 True。
    """
    # 固定 Python 内置的随机数生成器种子
    random.seed(seed)

    # 固定 NumPy 随机数生成器种子
    np.random.seed(seed)

    # 固定 PyTorch 随机数生成器种子（CPU）
    torch.manual_seed(seed)

    # 如果有 GPU，可固定其随机数生成器种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 所有 GPU 使用相同的种子

    # 是否启用确定性算法
    # if deterministic:
    #     # 强制 PyTorch 使用确定性算法
    #     torch.use_deterministic_algorithms(True)
    #
    #     # 设置 CuDNN 为确定性模式
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #
    #     # 配置 CUBLAS 确定性（对于 CUDA >= 10.2）
    #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #
    # print(f"Random seed set to {seed} and deterministic behavior {'enabled' if deterministic else 'disabled'}.")


def parse_option():
    # hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', type=int, default=56, help='local_rank')

    parser.add_argument('--model', type=str, default='super_net_x10p',  help='model name')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--batch_queue_size', type=int, default=50, help='batch_queue_size')
    parser.add_argument('--num_reader', type=int, default=2, help='num_reader')
    parser.add_argument('--max_steps', type=int, default=1000000, help='number of training steps')
    parser.add_argument('--log_freq', type=int, default=100, help='log frequency')
    parser.add_argument('--save_model_freq', type=int, default=5000, help='save frequency')

    parser.add_argument('--train_data_dir', type=str, default='train_frames',  help='train dataset dir')
    parser.add_argument('--valid_data_dir', type=str, default='valid_frames',  help='valid dataset dir')

    # model init
    parser.add_argument('--use_init_model', type=int, default=0, help='whether to load the model for initialization')
    parser.add_argument('--init_model_path', type=str, default='', help='init model path')

    # optimization
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--use_lr_decay', type=int, default=0, help='whether to use learning rate decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--use_grad_clip', type=int, default=1, help='use_grad_clip')
    parser.add_argument('--grad_clip_range', type=float, default=5.0, help='grad_clip_range')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    # trail
    parser.add_argument('--trial', type=str, default='test', help='trial name')

    # parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    # parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    # parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    opt = parser.parse_args()

    return opt


def main():
    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    opt = parse_option()
    set_seed(opt.seed)

    cwd = os.getcwd()
    logger.info(f"work dir: {cwd}")
    project_dir = ''
    if 'my_aiarena' in cwd:
        project_dir = os.path.join(cwd.split('my_aiarena')[0], "my_aiarena")
    opt.project_dir = project_dir
    trial_save_dir = os.path.join(project_dir, "output", opt.trial)
    opt.save_model_dir = os.path.join(trial_save_dir, "ckpt")
    opt.save_npz_log_dir = os.path.join(trial_save_dir, "npz_logs")
    opt.log_dir = os.path.join(trial_save_dir, "logs")

    os.makedirs(opt.save_model_dir, exist_ok=True)
    os.makedirs(opt.save_npz_log_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)



    opt.train_data_dir = os.path.join(project_dir, "dataset", opt.train_data_dir)
    opt.valid_data_dir = os.path.join(project_dir, "dataset", opt.valid_data_dir)

    trainer = Trainer(opt)
    trainer.run()


if __name__ == '__main__':
    main()
