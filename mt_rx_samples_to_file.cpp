//
// Copyright 2010-2011,2014 Ettus Research LLC
// Copyright 2018 Ettus Research, a National Instruments Company
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#include <uhd/exception.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/thread.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/program_options.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <complex>
#include <csignal>
#include <iostream>
#include <thread>

#include "sample_pipeline.h"
#include "sample_writer.h"

namespace po = boost::program_options;

static size_t nfft = 0, nfft_overlap = 0, nfft_div = 0, nfft_ds = 0, batches = 0, sample_id = 0;


static bool stop_signal_called = false;
void sig_int_handler(int)
{
    stop_signal_called = true;
}


void recv_to_file(uhd::usrp::multi_usrp::sptr usrp,
    const std::string& type,
    const std::string& wire_format,
    const size_t& channel,
    const std::string& file,
    const std::string& fft_file,
    size_t rate,
    size_t samps_per_buff,
    size_t zlevel,
    unsigned long long num_requested_samples,
    double time_requested       = 0.0,
    bool useVkFFT               = false)
{
    std::string cpu_format;
    set_sample_pipeline_types(type, cpu_format);

    uhd::stream_args_t stream_args(cpu_format, wire_format);
    std::vector<size_t> channel_nums;
    channel_nums.push_back(channel);
    stream_args.channels             = channel_nums;
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);

    const size_t max_samps_per_packet = rx_stream->get_max_num_samps();
    const size_t max_samples = std::max(max_samps_per_packet, samps_per_buff);
    std::cerr << "max_samps_per_packet from stream: " << max_samps_per_packet << std::endl;

    if (nfft) {
        std::cerr << "using FFT point size " << nfft << std::endl;

        if (samps_per_buff % nfft) {
            throw std::runtime_error("FFT point size must be a factor of spb");
        }
    }

    sample_pipeline_start(file, fft_file, max_samples, zlevel, useVkFFT, nfft, nfft_overlap, nfft_div, nfft_ds, rate, batches, sample_id);

    // setup streaming
    uhd::stream_cmd_t stream_cmd((num_requested_samples == 0)
                                     ? uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS
                                     : uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.num_samps  = size_t(num_requested_samples);
    stream_cmd.stream_now = true;
    stream_cmd.time_spec  = uhd::time_spec_t();

    uhd::rx_metadata_t md;
    size_t write_ptr = 0;
    size_t buffer_capacity = 0;
    bool overflow_message = true;
    size_t overflows = 0;
    unsigned long long num_total_samps = 0;

    const auto stop_time =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(int64_t(1000 * time_requested));
    rx_stream->issue_stream_cmd(stream_cmd);

    // Run this loop until either time expired (if a duration was given), until
    // the requested number of samples were collected (if such a number was
    // given), or until Ctrl-C was pressed.
    for (; not stop_signal_called
           and (num_requested_samples != num_total_samps or num_requested_samples == 0)
           and (time_requested == 0.0 or std::chrono::steady_clock::now() < stop_time);) {
        const auto now = std::chrono::steady_clock::now();
        char *buffer_p = get_sample_buffer(write_ptr, &buffer_capacity);
        size_t num_rx_samps =
            rx_stream->recv(buffer_p, max_samples, md, 3.0, false);

        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
            std::cerr << boost::format("Timeout while streaming") << std::endl;
            break;
        }
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
            ++overflows;
            if (overflow_message) {
                overflow_message = false;
                std::cerr << "Overflow!" << std::endl;
            }
            continue;
        }
        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
            std::string error = str(boost::format("Receiver error: %s") % md.strerror());
            throw std::runtime_error(error);
        }

        num_total_samps += num_rx_samps;
        size_t samp_bytes = num_rx_samps * get_samp_size();
        if (samp_bytes != buffer_capacity) {
            std::cerr << "resize to " << samp_bytes << " from " << buffer_capacity << std::endl;
            set_sample_buffer_capacity(write_ptr, samp_bytes);
        }

        enqueue_samples(write_ptr);
    }

    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    rx_stream->issue_stream_cmd(stream_cmd);
    std::cerr << "stream stopped" << std::endl;
    sample_pipeline_stop(overflows);
    std::cerr << "pipeline stopped" << std::endl;
}

bool check_locked_sensor(std::vector<std::string> sensor_names,
    const char* sensor_name,
    std::function<uhd::sensor_value_t(const std::string&)> get_sensor_fn,
    double setup_time)
{
    if (std::find(sensor_names.begin(), sensor_names.end(), sensor_name)
        == sensor_names.end())
        return false;

    auto setup_timeout = std::chrono::steady_clock::now()
                         + std::chrono::milliseconds(int64_t(setup_time * 1000));
    bool lock_detected = false;

    std::cerr << boost::format("Waiting for \"%s\": ") % sensor_name;
    std::cerr.flush();

    while (true) {
        if (lock_detected and (std::chrono::steady_clock::now() > setup_timeout)) {
            std::cerr << " locked." << std::endl;
            break;
        }
        if (get_sensor_fn(sensor_name).to_bool()) {
            std::cerr << "+";
            std::cerr.flush();
            lock_detected = true;
        } else {
            if (std::chrono::steady_clock::now() > setup_timeout) {
                std::cerr << std::endl;
                throw std::runtime_error(
                    str(boost::format(
                            "timed out waiting for consecutive locks on sensor \"%s\"")
                        % sensor_name));
            }
            std::cerr << "_";
            std::cerr.flush();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cerr << std::endl;
    return true;
}

int UHD_SAFE_MAIN(int argc, char* argv[])
{
    // variables to be set by po
    std::string args, file, fft_file, type, ant, subdev, ref, wirefmt;
    size_t channel, total_num_samps, spb, zlevel, rate;
    double option_rate, freq, gain, bw, total_time, setup_time, lo_offset;

    // setup the program options
    po::options_description desc("Allowed options");
    // clang-format off
    desc.add_options()
        ("help", "help message")
        ("args", po::value<std::string>(&args)->default_value(""), "multi uhd device address args")
        ("file", po::value<std::string>(&file)->default_value("usrp_samples.dat.zst"), "name of the file to write binary samples to")
        ("type", po::value<std::string>(&type)->default_value("short"), "sample type: double, float, or short")
        ("nsamps", po::value<size_t>(&total_num_samps)->default_value(0), "total number of samples to receive")
        ("duration", po::value<double>(&total_time)->default_value(0), "total number of seconds to receive")
        ("zlevel", po::value<size_t>(&zlevel)->default_value(1), "default compression level")
        ("spb", po::value<size_t>(&spb)->default_value(0), "samples per buffer (if 0, same as rate)")
        ("rate", po::value<double>(&option_rate)->default_value(20 * (1024*1024)), "rate of incoming samples")
        ("freq", po::value<double>(&freq)->default_value(100e6), "RF center frequency in Hz")
        ("lo-offset", po::value<double>(&lo_offset)->default_value(0.0),
            "Offset for frontend LO in Hz (optional)")
        ("gain", po::value<double>(&gain), "gain for the RF chain")
        ("ant", po::value<std::string>(&ant), "antenna selection")
        ("subdev", po::value<std::string>(&subdev), "subdevice specification")
        ("channel", po::value<size_t>(&channel)->default_value(0), "which channel to use")
        ("bw", po::value<double>(&bw), "analog frontend filter bandwidth in Hz")
        ("ref", po::value<std::string>(&ref)->default_value("internal"), "reference source (internal, external, mimo)")
        ("wirefmt", po::value<std::string>(&wirefmt)->default_value("sc16"), "wire format (sc8, sc16)")
        ("setup", po::value<double>(&setup_time)->default_value(1.0), "seconds of setup time")
        ("null", "run without writing to file")
        ("fftnull", "run without writing to FFT file")
        ("skip-lo", "skip checking LO lock status")
        ("int-n", "tune USRP with integer-N tuning")
        ("nfft", po::value<size_t>(&nfft)->default_value(0), "if > 0, calculate n FFT points")
        ("nfft_overlap", po::value<size_t>(&nfft_overlap)->default_value(0), "FFT overlap")
        ("nfft_div", po::value<size_t>(&nfft_div)->default_value(50), "calculate FFT over sample rate / n samples (e.g 50 == 20ms)")
        ("nfft_ds", po::value<size_t>(&nfft_ds)->default_value(10), "NFFT downsampling interval")
        ("fft_file", po::value<std::string>(&fft_file)->default_value(""), "name of file to write FFT points to (default derive from --file)")
        ("novkfft", "do not use vkFFT (use software FFT)")
        ("vkfft_batches", po::value<size_t>(&batches)->default_value(100), "vkFFT batches")
        ("vkfft_sample_id", po::value<size_t>(&sample_id)->default_value(0), "vkFFT sample_id")
    ;
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // print the help message
    if (vm.count("help")) {
        std::cerr << boost::format("UHD RX samples to file %s") % desc << std::endl;
        std::cerr << std::endl
                  << "This application streams data from a single channel of a USRP "
                     "device to a file.\n"
                  << std::endl;
        return ~0;
    }

    bool null                   = vm.count("null") > 0;
    bool fftnull                = vm.count("fftnull") > 0;
    bool useVkFFT               = vm.count("novkfft") == 0;

    if (!boost::algorithm::starts_with(wirefmt, "sc")) {
        throw std::runtime_error("non-complex wirefmt not supported");
    }

    if (option_rate <= 0.0) {
        throw std::runtime_error("invalid sample rate");
    }
    rate = size_t(option_rate);

    if (rate % nfft_div) {
        throw std::runtime_error("nfft_div must be a factor of sample rate");
    }

    if (spb == 0) {
        spb = rate;
        std::cerr << "defaulting spb to rate (" << spb << ")" << std::endl;
    }

    if (!fft_file.size()) {
        fft_file = get_prefix_file(file, "fft_");
    }

    if (null) {
        file.clear();
    }

    if (fftnull) {
        fft_file.clear();
    }

    // create a usrp device
    std::cerr << std::endl;
    std::cerr << boost::format("Creating the usrp device with: %s...") % args
              << std::endl;
    uhd::usrp::multi_usrp::sptr usrp = uhd::usrp::multi_usrp::make(args);

    // Lock mboard clocks
    if (vm.count("ref")) {
        usrp->set_clock_source(ref);
    }

    // always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("subdev"))
        usrp->set_rx_subdev_spec(subdev);

    std::cerr << boost::format("Using Device: %s") % usrp->get_pp_string() << std::endl;
    std::cerr << boost::format("Setting RX Rate: %f Msps...") % (rate / 1e6) << std::endl;
    usrp->set_rx_rate(rate, channel);
    std::cerr << boost::format("Actual RX Rate: %f Msps...")
                     % (usrp->get_rx_rate(channel) / 1e6)
              << std::endl
              << std::endl;

    // set the center frequency
    if (vm.count("freq")) { // with default of 0.0 this will always be true
        std::cerr << boost::format("Setting RX Freq: %f MHz...") % (freq / 1e6)
                  << std::endl;
        std::cerr << boost::format("Setting RX LO Offset: %f MHz...") % (lo_offset / 1e6)
                  << std::endl;
        uhd::tune_request_t tune_request(freq, lo_offset);
        if (vm.count("int-n"))
            tune_request.args = uhd::device_addr_t("mode_n=integer");
        usrp->set_rx_freq(tune_request, channel);
        std::cerr << boost::format("Actual RX Freq: %f MHz...")
                         % (usrp->get_rx_freq(channel) / 1e6)
                  << std::endl
                  << std::endl;
    }

    // set the rf gain
    if (vm.count("gain")) {
        std::cerr << boost::format("Setting RX Gain: %f dB...") % gain << std::endl;
        usrp->set_rx_gain(gain, channel);
        std::cerr << boost::format("Actual RX Gain: %f dB...")
                         % usrp->get_rx_gain(channel)
                  << std::endl
                  << std::endl;
    }

    // set the IF filter bandwidth
    if (vm.count("bw")) {
        std::cerr << boost::format("Setting RX Bandwidth: %f MHz...") % (bw / 1e6)
                  << std::endl;
        usrp->set_rx_bandwidth(bw, channel);
        std::cerr << boost::format("Actual RX Bandwidth: %f MHz...")
                         % (usrp->get_rx_bandwidth(channel) / 1e6)
                  << std::endl
                  << std::endl;

    // set the antenna
    if (vm.count("ant"))
        usrp->set_rx_antenna(ant, channel);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(int64_t(1000 * setup_time)));

    // check Ref and LO Lock detect
    if (not vm.count("skip-lo")) {
        check_locked_sensor(usrp->get_rx_sensor_names(channel),
            "lo_locked",
            [usrp, channel](const std::string& sensor_name) {
                return usrp->get_rx_sensor(sensor_name, channel);
            },
            setup_time);
        if (ref == "mimo") {
            check_locked_sensor(usrp->get_mboard_sensor_names(0),
                "mimo_locked",
                [usrp](const std::string& sensor_name) {
                    return usrp->get_mboard_sensor(sensor_name);
                },
                setup_time);
        }
        if (ref == "external") {
            check_locked_sensor(usrp->get_mboard_sensor_names(0),
                "ref_locked",
                [usrp](const std::string& sensor_name) {
                    return usrp->get_mboard_sensor(sensor_name);
                },
                setup_time);
        }
    }

    if (total_num_samps == 0) {
        std::signal(SIGINT, &sig_int_handler);
        std::cerr << "Press Ctrl + C to stop streaming..." << std::endl;
    }

    recv_to_file(usrp, type, wirefmt, channel, file, fft_file, rate, spb, zlevel, total_num_samps, total_time, useVkFFT);

    return EXIT_SUCCESS;
}
