# mt_rx_samples_to_file

enhanced UHD rx_samples_to_file

Uses background thread to write (optionally compressed) samples to storage, for CPU contrained platforms like Pi4.

See https://files.ettus.com/manual/page_transport.html for notes on tuning USRP transport options.

## example usage

Examples tuned for USB/B200 (https://github.com/EttusResearch/uhd/blob/master/host/lib/usrp/b200/b200_impl.hpp)

```
$ UHD_IMAGES_DIR=/usr/share/uhd/images mt_rx_samples_to_file --args num_recv_frames=128,recv_frame_size=16360 --file test.gz --nsamps 200000000 --rate 20000000 --freq 101e6 --spb 20000000
```

## build

```
$ mkdir build && cd build && cmake .. && make
```
