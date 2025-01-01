import argparse
import logging
import os

from torch import nn

from data.dataset import HoK3v3Dataset
from loss_fn import DistillKL
from trainer.trainer import Trainer


def parse_option():
    # hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_prefetch_worker', type=int, default=8, help='num_prefetch_worker')
    # parser.add_argument('--prefetch_factor', type=int, default=10, help='prefetch_factor')
    # parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--max_steps', type=int, default=1000000, help='number of training steps')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_log_freq', type=int, default=100, help='tensorboard log frequency')
    parser.add_argument('--save_model_freq', type=int, default=10000, help='save frequency')

    parser.add_argument('--data_dir', type=str, default='train_with_logits',
                        choices=['train_with_logits', 'train_frames'], help='dataset dir')

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

    opt = parse_option()

    cwd = os.getcwd()
    logging.info(f"work dir: {cwd}")
    if 'my_aiarena' in cwd:
        project_dir = os.path.join(cwd.split('my_aiarena')[0], "my_aiarena")
    trial_save_dir = os.path.join(project_dir, "output", opt.trial)
    opt.save_model_dir = os.path.join(trial_save_dir, "ckpt")
    opt.log_dir = os.path.join(trial_save_dir, "logs")

    os.makedirs(opt.save_model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    dataset_dir = os.path.join(project_dir, "dataset", opt.data_dir)
    logging.info(f"dataset dir: {dataset_dir}")
    train_dataset = HoK3v3Dataset(root_dir=dataset_dir,
                                  batch_size=opt.batch_size,
                                  sampling='random',
                                  sample_repeat_rate=1,
                                  chunk_size=10,
                                  batch_queue_size=20,
                                  npz_queue_size=10,
                                  num_sampler=2,
                                  num_reader=2
                                  )
    test_dataset = None

    from models.final_v1 import NetworkModel
    model = NetworkModel()

    criterion_dict = nn.ModuleDict({})
    criterion_dict['cls'] = nn.CrossEntropyLoss()
    criterion_dict['kd'] = DistillKL(opt.kd_T)

    trainer = Trainer(model,
                      criterion_dict,
                      train_dataset,
                      test_dataset,
                      opt)
    trainer.run()


if __name__ == '__main__':
    main()
