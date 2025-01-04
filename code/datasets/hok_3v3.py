import logging
import os
import time
from multiprocessing import Process, Queue, Value

import numpy as np
import torch


def _format_frame_x1_data(batch_data):
    """
    Formats batch data into state, action labels, and action logits.

    Args:
        batch_data (numpy.ndarray): Input data with shape (bs, 3, 4921).

    Returns:
        tuple: state, action_label, action_logits tensors.
    """
    # Convert numpy array to PyTorch tensor and rearrange dimensions
    batch = torch.from_numpy(batch_data).permute(1, 0, 2)  # Shape: (3, bs, 4921)

    # Define dimensions for splitting
    action_dims = sum([13, 25, 42, 42, 39])  # Total action dimensions: 161
    split_dims = [4586, action_dims, 1, 1, 5, action_dims, 1, 5]

    # Split batch into components
    (
        state,  # (3, bs, 4586)
        legal_action,
        reward,
        advantage,
        action_label,  # (3, bs, 5)
        action_logits,  # (3, bs, 161)
        is_train,
        sub_action_mask,
    ) = batch.split(split_dims, dim=-1)

    # Return the necessary components
    return state, action_label, action_logits, sub_action_mask


class NpzReader(Process):
    def __init__(self,
                 npz_files,
                 batch_size,
                 batch_queue,
                 counter,
                 ):
        super(NpzReader, self).__init__()
        self.npz_files = npz_files
        self.batch_size = batch_size
        self.chunk_size = self.batch_size // 512
        self.npz_read_idx = 0
        self.batch_queue = batch_queue
        self.generation_counter = counter
        self.logger = logging.getLogger(__name__)

    def _load_npz_file(self, file_path):
        data = np.load(file_path)
        # logging.debug(f'loaded file {file_path}')
        return data['samples']

    def run(self):
        while True:
            while not self.batch_queue.full():
                samples_list = []
                for _ in range(self.chunk_size):
                    if self.npz_read_idx == 0:
                        # 每次从头开始读取时，先打乱一次顺序
                        np.random.shuffle(self.npz_files)

                    samples = self._load_npz_file(self.npz_files[self.npz_read_idx])
                    samples_list.append(samples)
                    self.npz_read_idx = (self.npz_read_idx + 1) % len(self.npz_files)

                new_batch = np.concatenate(samples_list, axis=0)
                self.batch_queue.put(new_batch)
                self.generation_counter.add(new_batch.shape[0])


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


class FrameX1:
    def __init__(self,
                 root_dir,
                 batch_size=32,
                 batch_queue_size=20,
                 num_reader=2,
                 ):
        """
        单帧形式的数据集
        Args:
            root_dir: 指定数据集所在目录，绝对路径
            batch_size: 抽样的批次大小，属于随机抽样
            batch_queue_size: 指定预取的batch队列的长度
            num_reader: 指定读取npz文件的进程数量
        """
        self.root_dir = root_dir
        self.npz_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.npz')]
        np.random.shuffle(self.npz_files)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"dataset dir: {self.root_dir}, has {len(self.npz_files)} files.")
        self.batch_size = batch_size
        assert self.batch_size % 512 == 0, 'the batch size must be evenly divided by 512'
        self.consumption_counter = 0
        self.generation_counter = Counter(0)

        self.batch_queue_size = batch_queue_size
        self.batch_queue = Queue(maxsize=self.batch_queue_size)

        npz_chunks = np.array_split(self.npz_files, num_reader)
        self.npz_readers = [NpzReader(npz_chunks[i],
                                      self.batch_size,
                                      self.batch_queue,
                                      self.generation_counter
                                      ) for i in range(num_reader)]
        self._initialize()
        self.start_time = time.time()

    def _initialize(self):
        for p in self.npz_readers:
            p.daemon = True  # 设置为守护进程
            p.start()
        self.logger.info(f"stared {len(self.npz_readers)} npz_readers.")

    def get_next_batch(self):
        if self.batch_queue.qsize() == 0:
            self.logger.info("The batch queue is empty. Wait for a batch to be generated. "
                             f"batch_queue.qsize() = {self.batch_queue.qsize()}.")

        batch_data = self.batch_queue.get()
        self.consumption_counter += batch_data.shape[0]
        return _format_frame_x1_data(batch_data)

    def get_rate(self, reset=True):
        total_time = time.time() - self.start_time
        generation_rate = self.generation_counter.get() / total_time
        consumption_rate = self.consumption_counter / total_time
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
    dataset = FrameX1(root_dir=data_dir,
                      batch_size=32 * 16,
                      batch_queue_size=20,
                      num_reader=2
                      )

    for i in range(1000000):
        start_time = time.time()
        dataset.get_next_batch()
        logging.debug(
            f"get batch {i}, "
            f"batch_queue len = {dataset.batch_queue.qsize()}.")
        if i % 10 == 0:
            generation_rate, consumption_rate = dataset.get_rate()
            logging.debug(f'G rate = {generation_rate}, C rate = {consumption_rate}')
        logging.debug(f"step time = {time.time() - start_time}")
