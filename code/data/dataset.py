import logging
import os
import time
from multiprocessing import Process, Queue, Value

import numpy as np
import torch


def _format_frame_x1_data(batch_data):
    batch = torch.from_numpy(batch_data)  # (bs, 3, 4921)
    batch = batch.permute(1, 0, 2)  # (3, bs, 4921)
    action_dims = sum([13, 25, 42, 42, 39])
    state, leagal_action, reward, advantage, action_label, action_logits, is_train, sub_action_mask = \
        batch.split([4586, action_dims, 1, 1, 5, action_dims, 1, 5], dim=-1)
    return state, action_label, action_logits


def _format_frame_x16_data(batch_data):  # ([bs, 242352])  batch_size = -1
    datas = torch.from_numpy(batch_data)  # ([bs, 242352])
    datas = datas.view(-1, 3, 80784)  # ([bs, 3, 80784])
    datas = datas.permute(1, 0, 2)  # ([3, bs, 80784])

    hero_sequence_data_split_shape = [78736, 1024, 1024]  # [78736, 1024, 1024]
    batch, _, _ = datas.split(hero_sequence_data_split_shape, dim=-1)  # ([3, bs, 78736])
    batch = batch.view(3, -1, 16, 4921)
    batch = batch.reshape(3, -1, 4921)

    action_dims = sum([13, 25, 42, 42, 39])
    state, leagal_action, reward, advantage, action_label, action_logits, is_train, sub_action_mask = \
        batch.split([4586, action_dims, 1, 1, 5, action_dims, 1, 5], dim=-1)
    state = state.reshape(3, -1, 4586)
    action_label = action_label.reshape(3, -1, 5)
    action_logits = action_logits.reshape(3, -1, action_dims)
    return state, action_label, action_logits


# # 从 block_buff 中取一个batch的数据
# def _fetch_batch(chunk, indices, samples_per_file):
#     new_batch = []
#     for idx in indices:
#         block_idx = idx // samples_per_file
#         sample_idx = idx % samples_per_file
#         new_batch.append(chunk[block_idx][sample_idx: sample_idx + 1])
#     logging.debug(f'fetched new batch, len = {len(new_batch)}')
#     return np.concatenate(new_batch, axis=0)
#
#
# # 从 prefetch_file_queue 中取一个 chunk
# def _fetch_block(prefetch_file_queue, num_files):
#     new_block = []
#     for idx in range(num_files):
#         new_block.append(prefetch_file_queue.popleft())
#     logging.debug(f'fetched new chunk, len = {len(new_block)}')
#     return new_block


class BatchSampler(Process):
    def __init__(self,
                 batch_queue,
                 npz_queue,
                 batch_size=256,
                 sampling='random',
                 sample_repeat_rate=5,
                 chunk_size=5,
                 ):
        super(BatchSampler, self).__init__()

        self.samples_per_file = 1000
        self.sample_repeat_rate = sample_repeat_rate
        self.batch_size = batch_size
        self.sampling = sampling
        self.chunk = None  # 一个块，保存一批npz文件中的samples
        self.chunk_size = chunk_size  # 每个块有多少个 npz 的 samples
        self.fetch_idx = 0
        self.batch_indices_list = []
        self.batch_queue = batch_queue
        self.npz_queue = npz_queue
        self.logger = logging.getLogger(__name__)

    def _sample_batch(self, indices):
        """
        从 chunk 中抽样一个 batch
        """
        new_batch = []
        for idx in indices:
            block_idx = idx // self.samples_per_file
            sample_idx = idx % self.samples_per_file
            new_batch.append(self.chunk[block_idx][sample_idx: sample_idx + 1])
        return np.concatenate(new_batch, axis=0)

    def run(self):
        while True:
            while not self.batch_queue.full():
                if self.chunk is None or self.fetch_idx >= len(self.batch_indices_list):
                    # 生成一个新的 chunk
                    self.chunk = []
                    for idx in range(self.chunk_size):
                        samples = self.npz_queue.get()
                        self.chunk.append(samples)
                    self.fetch_idx = 0
                    self.samples_per_file = self.chunk[0].shape[0]
                    num_samples = len(self.chunk) * self.samples_per_file

                    # 生成待抽样的样本索引序列
                    if self.sampling == 'random':
                        sample_idx_list = list(np.random.permutation(num_samples)) * self.sample_repeat_rate
                        # sample_idx_list = np.array(
                        #     [np.random.permutation(num_samples) for _ in range(self.sample_repeat_rate)]
                        # ).reshape(-1).tolist()
                    else:
                        sample_idx_list = list(np.arange(num_samples)) * self.sample_repeat_rate
                    # 可被 batch_size 整除的长度，丢弃尾部不足 batch_size 的一个batch
                    truncated_length = (len(sample_idx_list) // self.batch_size) * self.batch_size
                    self.batch_indices_list = [sample_idx_list[i:i + self.batch_size] for i in
                                               range(0, truncated_length, self.batch_size)]

                indices = self.batch_indices_list[self.fetch_idx]
                new_batch = self._sample_batch(indices)
                self.batch_queue.put(new_batch)
                self.fetch_idx += 1


class NpzReader(Process):
    def __init__(self,
                 npz_files,
                 npz_queue,
                 counter,
                 ):
        super(NpzReader, self).__init__()
        self.npz_files = npz_files
        self.npz_read_idx = 0
        self.npz_queue = npz_queue
        self.generation_counter = counter
        self.logger = logging.getLogger(__name__)

    def _load_npz_file(self, file_path):
        data = np.load(file_path)
        # logging.debug(f'loaded file {file_path}')
        return data['samples']

    def run(self):
        while True:
            while not self.npz_queue.full():
                if self.npz_read_idx >= len(self.npz_files):
                    self.npz_read_idx = 0
                if self.npz_read_idx == 0:
                    # 每次从头开始读取时，先打乱一次顺序
                    np.random.shuffle(self.npz_files)

                samples = self._load_npz_file(self.npz_files[self.npz_read_idx])
                self.npz_queue.put(samples)
                self.generation_counter.add(samples.shape[0])
                self.npz_read_idx += 1


class Counter:
    def __init__(self, init_value=0):
        self.value = Value('L', init_value)  # 创建一个 unsigned long 共享整数，初始值为 0

    def add(self, num):
        with self.value.get_lock():
            self.value.value += num

    def reset(self):
        with self.value.get_lock():
            self.value.value = 0

    def get(self):
        return self.value.value


class HoK3v3Dataset:
    def __init__(self,
                 root_dir,
                 batch_size=32,
                 sampling='random',
                 sample_repeat_rate=1,
                 chunk_size=2,
                 batch_queue_size=20,
                 npz_queue_size=10,
                 num_sampler=2,
                 num_reader=2,
                 ):
        """
        采用流水线的方式读取数据集中的样本，并生成 batch。
        npz_queue 表示存放 npz 文件中的 samples 的队列，进程安全。
        batch_queue 是存放用于训练的 batch 的队列，进程安全。
        1. NpzReader 负责预取 npz 文件中的 samples 放入 npz_queue。
            当 npz_queue 不满时，NpzReader 负责不停的读取 npz 文件中的samples 放入 npz_queue。
        2. BatchSampler 负责预取 batch 放入 batch_queue，
            batch_queue 不满时，BatchSampler 先从 npz_queue 读取一个 chunk，
            然后从 chunk 中不断的抽样 batch，放入 batch_queue 中。

        Args:
            root_dir: 指定数据集所在目录，绝对路径
            batch_size: 抽样的批次大小
            sampling: 抽样方式，random 表示随机抽样，none 表示顺序抽样
            sample_repeat_rate: 样本重复使用的比率
            chunk_size: 指定抽样所在的块的大小，chunk_size=4表示在4个npz文件上抽样
            batch_queue_size: 指定预取的batch队列的长度
            npz_queue_size: 指定预取的npz文件队列的长度
            num_sampler: 指定抽样的进程数
            num_reader: 指定读取npz文件的进程数量
        """
        self.root_dir = root_dir
        self.npz_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.npz')]
        self.batch_size = batch_size
        self.single_frame = root_dir.endswith('frames')
        self.logger = logging.getLogger(__name__)
        self.consumption_counter = 0
        self.generation_counter = Counter(0)

        self.batch_queue_size = batch_queue_size
        self.npz_queue_size = npz_queue_size
        self.batch_queue = Queue(maxsize=self.batch_queue_size)
        self.npz_queue = Queue(maxsize=self.npz_queue_size)

        self.sampling = sampling
        self.sample_repeat_rate = sample_repeat_rate
        self.chunk_size = chunk_size
        self.batch_samplers = [BatchSampler(self.batch_queue,
                                            self.npz_queue,
                                            batch_size=self.batch_size,
                                            sampling=self.sampling,
                                            sample_repeat_rate=self.sample_repeat_rate,
                                            chunk_size=self.chunk_size,
                                            ) for _ in range(num_sampler)]
        npz_chunks = np.array_split(self.npz_files, num_reader)
        self.npz_readers = [NpzReader(npz_chunks[i],
                                      self.npz_queue,
                                      self.generation_counter
                                      ) for i in range(num_reader)]
        self._initialize()
        self.start_time = time.time()

    def _initialize(self):
        for p in self.npz_readers:
            p.daemon = True  # 设置为守护进程
            p.start()
        for p in self.batch_samplers:
            p.daemon = True  # 设置为守护进程
            p.start()
        self.logger.debug("stared npz_readers and batch_samplers.")

        # while self.batch_queue.qsize() <= 1:
        #     time.sleep(5)
        #     self.logger.debug("initializing, "
        #                       f"batch_queue.qsize() = {self.batch_queue.qsize()}, "
        #                       f"npz_queue.qsize() = {self.npz_queue.qsize()}")

    def get_next_batch(self):
        if self.batch_queue.qsize() == 0:
            self.logger.debug("The batch queue is empty. Wait for a batch to be generated. "
                              f"batch_queue.qsize() = {self.batch_queue.qsize()}, "
                              f"npz_queue.qsize() = {self.npz_queue.qsize()}.")

        batch_data = self.batch_queue.get()
        self.consumption_counter += batch_data.shape[0]
        return _format_frame_x1_data(batch_data) if self.single_frame else _format_frame_x16_data(batch_data)

    def get_rate(self, reset=True):
        total_time = time.time() - self.start_time
        generation_rate = self.generation_counter.get() / total_time
        consumption_rate = self.consumption_counter / total_time
        # self.logger.debug(f'G rate = {generation_rate}, C rate = {consumption_rate}')
        if reset:
            self.consumption_counter = 0
            self.generation_counter.reset()
            self.start_time = time.time()
        return generation_rate, consumption_rate


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(threadName)s-%(processName)s-%(name)s-%(filename)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )

    data_dir = os.path.join('/mnt/storage/yxh/match24/lightweight/my_aiarena', 'dataset', 'train_with_logits')
    dataset = HoK3v3Dataset(root_dir=data_dir,
                               batch_size=32,
                               sample_repeat_rate=1,
                               )

    for i in range(1000000):
        start_time = time.time()
        state, action_label, action_logits = dataset.get_next_batch()
        logging.debug(
            f"get batch {i}, "
            f"batch_queue len = {dataset.batch_queue.qsize()}, "
            f"npz_queue len = {dataset.npz_queue.qsize()}")
        if i % 10 == 0:
            generation_rate, consumption_rate = dataset.get_rate()
            logging.debug(f'G rate = {generation_rate}, C rate = {consumption_rate}')
        logging.debug(f"step time = {time.time() - start_time}")
