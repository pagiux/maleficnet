import math
import time
import hashlib

import torch
import numpy as np

from typing import Optional

from tqdm import tqdm

from pathlib import Path
from utils.utils_bit import bits_from_file, bits_from_bytes

from pyldpc import make_ldpc, encode


class Injector:
    BIT_TO_SIGNAL_MAPPING = {
        1: -1,
        0: 1
    }
    CHUNK_SIZE = 4096

    def __init__(self, seed: int, device: str, malware_path: Path, result_path: Path, logger, chunk_factor: int):
        self.seed = seed
        self.device = device
        self.malware_path = malware_path
        self.result_path = result_path
        self.payload = bits_from_file(malware_path)
        hash_str = hashlib.sha256(
            ''.join(str(l) for l in self.payload).encode('utf-8')).hexdigest()
        self.hash = bits_from_bytes(
            [char for char in hash_str.encode('utf-8')])
        self.message = self.payload + self.hash
        self.logger = logger
        self.chunk_factor = chunk_factor
        self.H = None
        self.G = None
        self.preamble = None
        if len(self.message) > 4000:
            k = 3048
        else:
            k = 96
        d_v = 3
        d_c = 12
        n = k * int(d_c / d_v)
        self.H, self.G = make_ldpc(
            n, d_v, d_c, systematic=True, sparse=True, seed=seed)

    def get_message_length(self, model):
        model_st_dict = model.state_dict()
        models_w = []
        layer_lengths = dict()

        layers = [n for n in model_st_dict.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x = model_st_dict[layer].detach().cpu().numpy().flatten()
            layer_lengths[layer] = len(x)
            models_w.extend(list(x))

        models_w = np.array(models_w)

        k = self.G.shape[1]

        snr1 = 10000000000000000
        c = []
        remaining_bits = len(self.message) % k
        n_chunks = int(len(self.message) / k)
        chunks = list()

        for ch in range(n_chunks):
            chunks.append(self.message[ch * k:ch * k + k])

        encoded = map(lambda x: encode(self.G, x, snr1), chunks)
        for enc in encoded:
            c.extend(enc)

        last_part = []
        last_part.extend(self.message[n_chunks * k:])
        last_part.extend([0] * (k - remaining_bits))

        c.extend(encode(self.G, last_part, snr1))

        np.random.seed(self.seed * 15)
        preamble = np.sign(np.random.uniform(-1, 1, 200))
        b = np.concatenate((preamble, c))

        return len(b)

    def inject(self, model, gamma: Optional[float] = None):
        start = time.time()

        model_st_dict = model.state_dict()
        models_w = []
        layer_lengths = dict()

        layers = [n for n in model_st_dict.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x = model_st_dict[layer].detach().cpu().numpy().flatten()
            layer_lengths[layer] = len(x)
            models_w.extend(list(x))

        models_w = np.array(models_w)

        k = self.G.shape[1]

        snr1 = 10000000000000000
        c = []
        remaining_bits = len(self.message) % k
        n_chunks = int(len(self.message) / k)
        chunks = list()

        for ch in range(n_chunks):
            chunks.append(self.message[ch * k:ch * k + k])

        encoded = map(lambda x: encode(self.G, x, snr1), chunks)
        for enc in encoded:
            c.extend(enc)

        last_part = []
        last_part.extend(self.message[n_chunks * k:])
        last_part.extend([0] * (k - remaining_bits))

        c.extend(encode(self.G, last_part, snr1))

        np.random.seed(self.seed * 15)
        preamble = np.sign(np.random.uniform(-1, 1, 200))
        b = np.concatenate((preamble, c))

        number_of_chunks = math.ceil(len(b) / self.CHUNK_SIZE)
        if self.CHUNK_SIZE * self.chunk_factor * number_of_chunks > len(models_w):
            self.logger.critical(
                f'Spreading codes cannot be bigger than the model!')
            return

        np.random.seed(self.seed)
        filter_indexes = np.random.randint(
            0, len(models_w), self.CHUNK_SIZE * self.chunk_factor * number_of_chunks, np.int32).tolist()

        self.logger.info(
            f'Injecting on {self.CHUNK_SIZE * self.chunk_factor} * {number_of_chunks} = {self.CHUNK_SIZE * self.chunk_factor * number_of_chunks} parameters')
        with tqdm(total=len(b)) as bar:
            bar.set_description('Injecting')
            current_chunk = 0
            current_bit = 0
            np.random.seed(self.seed)
            for bit in b:
                spreading_code = np.random.choice(
                    [-1, 1], size=self.CHUNK_SIZE * self.chunk_factor)
                current_bit_cdma_signal = gamma * spreading_code * bit
                current_filter_index = filter_indexes[current_chunk * self.CHUNK_SIZE * self.chunk_factor:
                                                      (current_chunk + 1) * self.CHUNK_SIZE * self.chunk_factor]
                models_w[current_filter_index] = np.add(
                    models_w[current_filter_index], current_bit_cdma_signal)

                current_bit += 1
                if current_bit > self.CHUNK_SIZE * (current_chunk + 1):
                    current_chunk += 1

                bar.update(1)

        curr_index = 0
        for layer in layers:
            x = np.array(
                models_w[curr_index:curr_index + layer_lengths[layer]])
            model_st_dict[layer] = torch.from_numpy(np.reshape(
                x, model_st_dict[layer].shape)).to(self.device)
            curr_index = curr_index + layer_lengths[layer]

        end = time.time()
        self.logger.info(f'Time to inject {end - start}')
        return model_st_dict, len(b), len(self.payload), len(self.hash)
