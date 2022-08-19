# mt_rx_samples_to_file

enhanced UHD rx_samples_to_file

Uses background thread to write (optionally compressed) samples to storage, for CPU contrained platforms like Pi4.

See https://files.ettus.com/manual/page_transport.html for notes on tuning USRP transport options.

## example usage

Examples tuned for USB/B200 (https://github.com/EttusResearch/uhd/blob/master/host/lib/usrp/b200/b200_impl.hpp)

```
$ UHD_IMAGES_DIR=/usr/share/uhd/images mt_rx_samples_to_file --args num_recv_frames=128,recv_frame_size=16360 --file test.gz --nsamps 200000000 --rate 20000000 --freq 101e6 --spb 20000000
```

## PSD

PSD using Welch's method can be simultaneously calculated and recorded.

In this example, 5 seconds of samples centered at 108MHz at 1.024Ms/s are recorded, and the results plotted in python.

```
$ mt_rx_samples_to_file --args num_recv_frames=128,recv_frame_size=16360 --freq 108e6 --rate 1.024e6 --gain 40 --duration 5 --nfft 2048 --file recording_1_108000000Hz_1024000sps.s16.zst
$ zstd -df fft_recording_1_108000000Hz_1024000sps.s16.zst
$ python3
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> NFFT=2048
>>> i = np.fromfile('fft_recording_1_108000000Hz_1024000sps.s16', dtype=np.float32)
>>> i = np.roll(i.reshape(-1, NFFT).swapaxes(0, 1), int(NFFT / 2), 0)
>>> plt.imshow(result, cmap='turbo')
>>> plt.show()
```

## build

```
$ mkdir build && cd build && cmake .. && make
```
