#!/usr/bin/python3

import zstandard
import numpy as np
import matplotlib.pyplot as plt

NFFT=2048

with open('fft_usrp_samples.dat.zst', 'rb') as f:
    with zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True) as reader:
        i = np.frombuffer(reader.read(), dtype=np.float32)
        i = np.roll(i.reshape(-1, NFFT).swapaxes(0, 1), int(NFFT / 2), 0)
        plt.imshow(i, cmap='turbo')
        plt.show()
