import math
import time
import hashlib
import numpy as np

import multiprocessing as mp

from tqdm import tqdm

from pathlib import Path
from utils.utils_bit import bits_to_file, bits_from_bytes

from pyldpc import make_ldpc, decode, get_message


_func = None


def worker_init(func):
  global _func
  _func = func


def worker(x):
  return _func(x)


class Extractor:
    BIT_TO_SIGNAL_MAPPING = {
        1: -1,
        0: 1
    }
    CHUNK_SIZE = 4096

    def __init__(self, seed: int, device: str, result_path: Path, logger, malware_length: int, hash_length: int, chunk_factor: int):
        self.seed = seed
        self.device = device
        self.result_path = result_path
        self.logger = logger
        self.H = None
        self.G = None
        self.preamble = None
        self.malware_length = malware_length
        self.hash_length = hash_length
        self.chunk_factor = chunk_factor
        if self.malware_length > 4000:
            k = 3048
        else:
            k = 96
        d_v = 3
        d_c = 12
        n = k * int(d_c / d_v)
        self.H, self.G = make_ldpc(
            n, d_v, d_c, systematic=True, sparse=True, seed=seed)

    def extract(self, model, message_length, malware_name):
        extraction_path = self.result_path
        extraction_path.mkdir(parents=True, exist_ok=True)

        start = time.time()
        st_dict_next = model.state_dict()

        models_w_curr = []

        layer_lengths = dict()
        total_params = 0

        layers = [n for n in st_dict_next.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x_curr = st_dict_next[layer].detach().cpu().numpy().flatten()
            models_w_curr.extend(list(x_curr))
            layer_lengths[layer] = len(x_curr)
            total_params += len(x_curr)

        models_w_curr = np.array(models_w_curr)

        number_of_chunks = math.ceil(message_length / self.CHUNK_SIZE)
        if self.CHUNK_SIZE * self.chunk_factor * number_of_chunks > len(models_w_curr):
            self.logger.critical(
                f'Spreading codes cannot be bigger than the model!')
            return

        np.random.seed(self.seed)
        filter_indexes = np.random.randint(
            0, len(models_w_curr), self.CHUNK_SIZE * self.chunk_factor * number_of_chunks, np.int32).tolist()

        x = []
        ys = []

        with tqdm(total=message_length) as bar:
            bar.set_description('Extracting')
            current_chunk = 0
            current_bit = 0
            np.random.seed(self.seed)
            for _ in range(message_length):
                spreading_code = np.random.choice(
                    [-1, 1], size=self.CHUNK_SIZE * self.chunk_factor)
                current_filter_index = filter_indexes[current_chunk * self.CHUNK_SIZE * self.chunk_factor:
                                                      (current_chunk + 1) * self.CHUNK_SIZE * self.chunk_factor]
                current_models_w_delta = models_w_curr[current_filter_index]
                y_i = np.matmul(spreading_code.T, current_models_w_delta)
                ys.append(y_i)

                current_bit += 1
                if current_bit > self.CHUNK_SIZE * (current_chunk + 1):
                    current_chunk += 1

                bar.update(1)

        y = np.array(ys)

        np.random.seed(self.seed * 15)
        preamble = np.sign(np.random.uniform(-1, 1, 200))

        gain = np.mean(np.multiply(y[:200], preamble))
        sigma = np.std(np.multiply(y[:200], preamble) / gain)
        snr = -20 * np.log10(sigma)
        self.logger.info(f'Signal to Noise Ratio = {snr}')

        k = self.G.shape[0]
        y = y[200:]
        n_chunks = int(len(y) / k)
        chunks = list()

        for ch in range(n_chunks):
            chunks.append(y[ch * k:ch * k + k] / gain)

        d = map(lambda x: decode(self.H, x, snr), chunks)

        self.logger.info(f'Starting a pool of {mp.cpu_count() - 3} processes to get the malware.')
        with mp.Pool(mp.cpu_count() - 3, initializer=worker_init, initargs=(lambda x: get_message(self.G, x),)) as pool:
            decoded = pool.map(worker, d)

        for dec in decoded:
            x.extend(dec)

        end = time.time()
        self.logger.info(f'Time to extract {end - start}')

        bits_to_file(extraction_path / f'{malware_name}.no_execute',
                     x[:self.malware_length])

        str_malware = ''.join(str(l) for l in x[:self.malware_length])
        str_hash = ''.join(
            str(l) for l in x[self.malware_length:self.malware_length+self.hash_length])
        hash_str = hashlib.sha256(
            ''.join(str(l) for l in str_malware).encode('utf-8')).hexdigest()
        hash_bits = ''.join(str(l) for l in (bits_from_bytes(
            [char for char in hash_str.encode('utf-8')])))
        self.logger.info(
            f'Original malware hash {str_hash}\nExtracted malware hash {hash_bits}')

        return str_hash == hash_bits
