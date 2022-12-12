# OBSOLETE

Please see https://github.com/iqtlabs/uhd_sample_recorder.

# mt_rx_samples_to_file

enhanced UHD rx_samples_to_file

Uses background thread to write (optionally compressed) samples to storage, for CPU contrained platforms like Pi4.

See https://files.ettus.com/manual/page_transport.html for notes on tuning USRP transport options.

## example usage

Examples tuned for USB/B200 (https://github.com/EttusResearch/uhd/blob/master/host/lib/usrp/b200/b200_impl.hpp)

```
$ UHD_IMAGES_DIR=/usr/share/uhd/images mt_rx_samples_to_file --args num_recv_frames=128,recv_frame_size=16360 --file test.gz --nsamps 200000000 --rate 20000000 --freq 101e6 --spb 20000000
```

## FFT support

Requires a Vulkan compatible GPU

### Running on a Raspberry Pi4

This works on both Raspberry Pi OS (Debian) and Ubuntu.

1. Add `dtoverlay=vc4-kms-v3d-pi4` to `/boot/firmware/config.txt` (`/boot/config.txt` on Raspberry Pi OS).
2. Make sure your user has access to the ```/dev/dri/renderD128``` device (e.g. ```sudo usermod -a -G render ubuntu```).
3. If running Ubuntu 22.04 or later, proceed to the [Build section](#build). If running Raspberry Pi OS, you will first need to update your Vulkan/MESA drivers - use https://github.com/jmcerrejon/PiKISS to Configure and install Vulkan using the main branch first then proceed to the [Build section](#build).

In this example, 5 seconds of samples centered at 108MHz at 1.024Ms/s are recorded, and the FFT results are plotted in python.

```
$ sudo ./build/mt_rx_samples_to_file --args num_recv_frames=128,recv_frame_size=16360 --freq 108e6 --rate 1.024e6 --gain 40 --duration 5 --nfft 2048
using vkFFT batch size 10 on V3D 4.2

Creating the usrp device with: num_recv_frames=128,recv_frame_size=16360...
[INFO] [UHD] linux; GNU C++ version 11.2.0; Boost_107400; UHD_4.1.0.5-3

[INFO] [B200] Loading firmware image: /usr/share/uhd/images/usrp_b200_fw.hex...
[INFO] [B200] Detected Device: B200
[INFO] [B200] Loading FPGA image: /usr/share/uhd/images/usrp_b200_fpga.bin...
[INFO] [B200] Operating over USB 3.
[INFO] [B200] Detecting internal GPSDO.... 
[INFO] [GPS] No GPSDO found
[INFO] [B200] Initialize CODEC control...
[INFO] [B200] Initialize Radio control...
[INFO] [B200] Performing register loopback test... 
[INFO] [B200] Register loopback test passed
[INFO] [B200] Setting master clock rate selection to 'automatic'.
[INFO] [B200] Asking for clock rate 16.000000 MHz... 
[INFO] [B200] Actually got clock rate 16.000000 MHz.
Using Device: Single USRP:
  Device: B-Series Device
  Mboard 0: B200
  RX Channel: 0
    RX DSP: 0
    RX Dboard: A
    RX Subdev: FE-RX1
  TX Channel: 0
    TX DSP: 0
    TX Dboard: A
    TX Subdev: FE-TX1

Setting RX Rate: 1.024000 Msps...
[INFO] [B200] Asking for clock rate 32.768000 MHz... 
[INFO] [B200] Actually got clock rate 32.768000 MHz.
Actual RX Rate: 1.024000 Msps...

Setting RX Freq: 108.000000 MHz...
Setting RX LO Offset: 0.000000 MHz...
Actual RX Freq: 108.000000 MHz...

Setting RX Gain: 40.000000 dB...
Actual RX Gain: 40.000000 dB...

Waiting for "lo_locked": ++++++++++ locked.

Press Ctrl + C to stop streaming...
defaulting spb to rate (1024000)
max_samps_per_packet from stream: 4086
max_buffer_size: 4096000 (1024000 samples)
opening /home/ubuntu/mt_rx_samples_to_file/.usrp_samples.dat.zst
writing zstd compressed output
using FFT point size 2048
opening /home/ubuntu/mt_rx_samples_to_file/.fft_usrp_samples.dat.zst
writing zstd compressed output
stream stopped
calls: 20480000
total: 20480000
closing usrp_samples.dat.zst
closing /home/ubuntu/mt_rx_samples_to_file/fft_usrp_samples.dat.zst
closed

Done!
$ ./plot_fft.py
```

## Build

```
sudo apt-get update && sudo apt-get install -qy build-essential cmake cppcheck libarmadillo-dev libboost-all-dev libuhd-dev libvulkan-dev python3-pip uhd-host unzip wget && \
        sudo pip3 install zstandard && \
        sudo /usr/lib/uhd/utils/uhd_images_downloader.py -t "b2|usb" && \
        wget https://sourceforge.net/projects/sigpack/files/sigpack-1.2.7.zip -O sigpack.zip && unzip sigpack.zip && ln -s sigpack-*/sigpack . && \
        git clone https://github.com/DTolm/VkFFT -b v1.2.31 && \
        wget https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp && \
        cd VkFFT && mkdir build && cd build && cmake .. && make -j $(nproc) && cd ../.. && \
        mkdir build && cd build && cmake .. && make && cd ..
```
