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
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zstd.hpp>
#include <boost/thread/thread.hpp>
#include <boost/atomic.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <complex>
#include <csignal>
#include <fstream>
#include <iostream>
#include <thread>

#include "sigpack/sigpack.h"
#include "vkFFT.h"
#include "utils_VkFFT.h"
#include "ShaderLang.h"

namespace po = boost::program_options;

const size_t kFFTbufferCount = 256;
const size_t kSampleBuffers = 16;

static boost::iostreams::filtering_streambuf<boost::iostreams::output> outbuf;
static boost::iostreams::filtering_streambuf<boost::iostreams::output> fft_outbuf;
static std::ostream out(&outbuf);
static std::ostream fft_out(&fft_outbuf);

static char* sampleBuffers[kSampleBuffers];
static size_t sampleBuffersCapacity[kSampleBuffers];
static arma::cx_fmat inFFTBuffers[kFFTbufferCount];
static arma::cx_fmat outFFTbuffers[kFFTbufferCount];
static boost::atomic<bool> samples_input_done(false);
static boost::atomic<bool> write_samples_worker_done(false);
static boost::atomic<bool> fft_in_worker_done(false);
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kSampleBuffers>> sample_queue;
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kFFTbufferCount>> in_fft_queue;
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kFFTbufferCount>> out_fft_queue;
static size_t nfft = 0, nfft_overlap = 0, nfft_div = 0, rate = 0, ffts_in = 0, ffts_out = 0;

static arma::fvec hammingWindow;
static float hammingWindowSum = 0;

typedef std::complex<int16_t> sample_t;

static bool useVkFFT = true;
const uint64_t kVkNumBuf = 1;
static VkGPU vkGPU = {};
static VkFFTConfiguration vkConfiguration = {};
static VkDeviceMemory* vkBufferDeviceMemory = NULL;
static VkFFTApplication vkApp = {};
static VkFFTLaunchParams vkLaunchParams = {};


VkFFTResult init_vkfft(size_t batches, size_t sample_id) {
    vkGPU.enableValidationLayers = 0;

    VkResult res = VK_SUCCESS;
    res = createInstance(&vkGPU, sample_id);
    if (res) {
	std::cout << "VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE" << std::endl;
	return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;
    }
    res = setupDebugMessenger(&vkGPU);
    if (res != 0) {
	//printf("Debug messenger creation failed, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER;
    }
    res = findPhysicalDevice(&vkGPU);
    if (res != 0) {
	//printf("Physical device not found, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
    }
    res = createDevice(&vkGPU, sample_id);
    if (res != 0) {
	//printf("Device creation failed, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;
    }
    res = createFence(&vkGPU);
    if (res != 0) {
	//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_CREATE_FENCE;
    }
    res = createCommandPool(&vkGPU);
    if (res != 0) {
	//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
	return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL;
    }
    vkGetPhysicalDeviceProperties(vkGPU.physicalDevice, &vkGPU.physicalDeviceProperties);
    vkGetPhysicalDeviceMemoryProperties(vkGPU.physicalDevice, &vkGPU.physicalDeviceMemoryProperties);

    glslang_initialize_process();//compiler can be initialized before VkFFT

    vkConfiguration.FFTdim = 1;
    vkConfiguration.size[0] = nfft;
    vkConfiguration.size[1] = 1;
    vkConfiguration.size[2] = 1;
    vkConfiguration.device = &vkGPU.device;
    vkConfiguration.queue = &vkGPU.queue;
    vkConfiguration.fence = &vkGPU.fence;
    vkConfiguration.commandPool = &vkGPU.commandPool;
    vkConfiguration.physicalDevice = &vkGPU.physicalDevice;
    vkConfiguration.isCompilerInitialized = true;
    vkConfiguration.doublePrecision = false;
    vkConfiguration.numberBatches = batches;

    std::cout << "using vkFFT batch size " << vkConfiguration.numberBatches << " on " << vkGPU.physicalDeviceProperties.deviceName << std::endl;

    vkConfiguration.bufferSize = (uint64_t*)aligned_alloc(sizeof(uint64_t), sizeof(uint64_t) * kVkNumBuf);
    if (!vkConfiguration.bufferSize) return VKFFT_ERROR_MALLOC_FAILED;

    const size_t buffer_size_f = vkConfiguration.size[0] * vkConfiguration.size[1] * vkConfiguration.size[2] * vkConfiguration.numberBatches / kVkNumBuf;
    for (uint64_t i = 0; i < kVkNumBuf; ++i) {
	vkConfiguration.bufferSize[i] = (uint64_t)sizeof(std::complex<float>) * buffer_size_f;
    }

    vkConfiguration.bufferNum = kVkNumBuf;

    vkConfiguration.buffer = (VkBuffer*)aligned_alloc(sizeof(std::complex<float>), kVkNumBuf * sizeof(VkBuffer));
    if (!vkConfiguration.buffer) return VKFFT_ERROR_MALLOC_FAILED;
    vkBufferDeviceMemory = (VkDeviceMemory*)aligned_alloc(sizeof(std::complex<float>), kVkNumBuf * sizeof(VkDeviceMemory));
    if (!vkBufferDeviceMemory) return VKFFT_ERROR_MALLOC_FAILED;

    for (uint64_t i = 0; i < kVkNumBuf; ++i) {
	vkConfiguration.buffer[i] = {};
	vkBufferDeviceMemory[i] = {};
	VkFFTResult resFFT = allocateBuffer(&vkGPU, &vkConfiguration.buffer[i], &vkBufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vkConfiguration.bufferSize[i]);
	if (resFFT != VKFFT_SUCCESS) return VKFFT_ERROR_MALLOC_FAILED;
    }

    vkLaunchParams.buffer = vkConfiguration.buffer;

    VkFFTResult resFFT = initializeVkFFT(&vkApp, vkConfiguration);
    if (resFFT != VKFFT_SUCCESS) return resFFT;

    return VKFFT_SUCCESS;
}


void free_vkfft() {
    for (uint64_t i = 0; i < kVkNumBuf; i++) {
	vkDestroyBuffer(vkGPU.device, vkConfiguration.buffer[i], NULL);
	vkFreeMemory(vkGPU.device, vkBufferDeviceMemory[i], NULL);
    }
    free(vkConfiguration.buffer);
    free(vkBufferDeviceMemory);
    free(vkConfiguration.bufferSize);
    deleteVkFFT(&vkApp);
}


void fft_out_offload(arma::cx_fmat &Pw) {
    ++ffts_out;
    Pw /= hammingWindowSum;
    // TODO: offload C2R
    // TOOD: write spectrogram as image.
    arma::fmat fft_points_out = log10(real(Pw % conj(Pw))) * 10;
    if (!fft_outbuf.empty()) {
	fft_out.write((const char*)fft_points_out.memptr(), fft_points_out.n_elem * sizeof(float));
    }
}


template <class T1>
void specgram(size_t &fft_write_ptr, const arma::Col<T1>& x, const arma::uword Nfft=512, const arma::uword Noverl=256)
{
    ++ffts_in;
    arma::uword N = x.size();
    arma::uword D = Nfft-Noverl;
    arma::uword m = 0;
    const arma::uword U = static_cast<arma::uword>(floor((N-Noverl)/double(D)));
    size_t row_size = Nfft * sizeof(T1);
    arma::cx_fmat &Pw_in = inFFTBuffers[fft_write_ptr];
    arma::cx_fmat &Pw = outFFTbuffers[fft_write_ptr];
    Pw_in.set_size(Nfft,U);
    Pw.copy_size(Pw_in);

    for(arma::uword k=0; k<=N-Nfft; k+=D)
    {
	Pw_in.col(m++) = x.rows(k,k+Nfft-1) % hammingWindow;
    }

    while (!in_fft_queue.push(fft_write_ptr)) {
        usleep(100);
    }

    if (++fft_write_ptr == kFFTbufferCount) {
	fft_write_ptr = 0;
    }
}


inline void write_samples(size_t &fft_write_ptr, arma::cx_fvec &fft_samples_in)
{
    size_t read_ptr;

    while (sample_queue.pop(read_ptr)) {
	char *buffer_p = sampleBuffers[read_ptr];
	size_t buffer_capacity = sampleBuffersCapacity[read_ptr];
	if (nfft) {
	    sample_t *i_p = (sample_t*) buffer_p;
	    for (size_t i = 0; i < buffer_capacity / (fft_samples_in.size() * sizeof(sample_t)); ++i) {
		for (size_t fft_p = 0; fft_p < fft_samples_in.size(); ++fft_p, ++i_p) {
		    fft_samples_in[fft_p] = std::complex<float>(i_p->real(), i_p->imag());
		}
		specgram(fft_write_ptr, fft_samples_in, nfft, nfft_overlap);
	    }
	}
	if (!outbuf.empty()) {
	    out.write((const char*)buffer_p, buffer_capacity);
	}
	std::cout << "." << std::endl;
    }
}


inline void fftout() {
    size_t read_ptr;
    while (out_fft_queue.pop(read_ptr)) {
	fft_out_offload(outFFTbuffers[read_ptr]);
    }
}


void fft_out_worker(void)
{
    while (!fft_in_worker_done) {
	fftout();
	usleep(10000);
    }
    fftout();
    std::cout << "fft out worker done" << std::endl;
}


void specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw) {
    for(arma::uword k=0; k < Pw_in.n_cols; ++k)
    {
	Pw.col(k) = arma::fft(Pw_in.col(k), nfft);
    }
}


void vkfft_specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw) {
    size_t row_size = Pw_in.n_rows * sizeof(std::complex<float>);

    // TODO: offload windowing.
    for(arma::uword k=0; k < Pw_in.n_cols; k += vkConfiguration.numberBatches)
    {
	size_t offset = k * row_size;
	// TODO: retain overlap buffer from previous spectrum() call.
	uint64_t buffer_size = std::min(vkConfiguration.bufferSize[0], (uint64_t)((Pw_in.n_cols - k) * row_size));
	transferDataFromCPU(&vkGPU, ((char*)Pw_in.memptr()) + offset, &vkConfiguration.buffer[0], buffer_size);
	performVulkanFFT(&vkGPU, &vkApp, &vkLaunchParams, -1, 1);
	transferDataToCPU(&vkGPU, ((char*)Pw.memptr()) + offset, &vkConfiguration.buffer[0], buffer_size);
    }
}


inline void fftin() {
    size_t read_ptr;
    while (in_fft_queue.pop(read_ptr)) {
	if (useVkFFT) {
	    vkfft_specgram_offload(inFFTBuffers[read_ptr], outFFTbuffers[read_ptr]);
	} else {
	    specgram_offload(inFFTBuffers[read_ptr], outFFTbuffers[read_ptr]);
	}
	while (!out_fft_queue.push(read_ptr)) {
            usleep(100);
	}
    }
}


void fft_in_worker(void)
{
    while (!write_samples_worker_done) {
	fftin();
	usleep(10000);
    }
    fftin();
    fft_in_worker_done = true;
    std::cout << "fft worker done" << std::endl;
}


void write_samples_worker(void)
{
    size_t fft_write_ptr = 0;
    arma::cx_fvec fft_samples_in(rate / nfft_div);

    while (!samples_input_done) {
	write_samples(fft_write_ptr, fft_samples_in);
	usleep(10000);
    }
    write_samples(fft_write_ptr, fft_samples_in);
    write_samples_worker_done = true;
    std::cout << "write samples worker done" << std::endl;
}


static bool stop_signal_called = false;
void sig_int_handler(int)
{
    stop_signal_called = true;
}


void open_samples(std::string &dotfile, boost::filesystem::path &orig_path, size_t zlevel,
		  std::ofstream *outfile_p, boost::iostreams::filtering_streambuf<boost::iostreams::output> *outbuf_p) {
    std::cout << "opening " << dotfile << std::endl;
    outfile_p->open(dotfile.c_str(), std::ofstream::binary);
    if (!outfile_p->is_open()) {
	throw std::runtime_error(dotfile + " could not be opened");
    }
    if (orig_path.has_extension()) {
	if (orig_path.extension() == ".gz") {
	    std::cout << "writing gzip compressed output" << std::endl;
	    outbuf_p->push(boost::iostreams::gzip_compressor(
		boost::iostreams::gzip_params(zlevel)));
	} else if (orig_path.extension() == ".zst") {
	    std::cout << "writing zstd compressed output" << std::endl;
	    outbuf_p->push(boost::iostreams::zstd_compressor(
		boost::iostreams::zstd_params(zlevel)));
	} else {
	    std::cout << "writing uncompressed output" << std::endl;
	}
    }
    outbuf_p->push(*outfile_p);
}


void close_samples(const std::string &file, std::string &dotfile, const std::string &dirname, size_t overflows,
		   std::ofstream *outfile_p, boost::iostreams::filtering_streambuf<boost::iostreams::output> *outbuf_p) {
    if (outfile_p->is_open()) {
	std::cout << "closing " << file << std::endl;
	boost::iostreams::close(*outbuf_p);
	outfile_p->close();

	if (overflows) {
	    std::string overflow_name = dirname + "/overflow-" + file;
	    rename(dotfile.c_str(), overflow_name.c_str());
	} else {
	    rename(dotfile.c_str(), file.c_str());
	}
    }
}

template <typename samp_type>
void recv_to_file(uhd::usrp::multi_usrp::sptr usrp,
    const std::string& cpu_format,
    const std::string& wire_format,
    const size_t& channel,
    const std::string& file,
    size_t samps_per_buff,
    size_t zlevel,
    unsigned long long num_requested_samples,
    double time_requested       = 0.0,
    bool bw_summary             = false,
    bool stats                  = false,
    bool null                   = false,
    bool fftnull                = false,
    bool enable_size_map        = false,
    bool continue_on_bad_packet = false)
{
    unsigned long long num_total_samps = 0;
    // create a receive streamer
    uhd::stream_args_t stream_args(cpu_format, wire_format);
    std::vector<size_t> channel_nums;
    channel_nums.push_back(channel);
    stream_args.channels             = channel_nums;
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);

    if (samps_per_buff == 0) {
	samps_per_buff = rate;
	std::cout << "defaulting spb to rate (" << samps_per_buff << ")" << std::endl;
    }

    const size_t max_samps_per_packet = rx_stream->get_max_num_samps();
    const size_t max_samples = std::max(max_samps_per_packet, samps_per_buff);
    std::cout << "max_samps_per_packet from stream: " << max_samps_per_packet << std::endl;
    static size_t max_buffer_size = max_samples * sizeof(samp_type);
    std::cout << "max_buffer_size: " << max_buffer_size << " (" << max_samples << " samples)" << std::endl;

    uhd::rx_metadata_t md;
    std::ofstream outfile;
    std::ofstream fft_outfile;
    boost::filesystem::path orig_path(file);
    std::string basename(orig_path.filename().c_str());
    std::string fft_basename = "fft_" + basename;
    std::string dirname(boost::filesystem::canonical(orig_path.parent_path()).c_str());
    std::string dotfile = dirname + "/." + basename;
    std::string fft_dotfile = dirname + "/." + fft_basename;
    std::string fft_file = dirname + "/" + fft_basename;

    for (size_t i = 0; i < kSampleBuffers; ++i) {
	sampleBuffersCapacity[i] = max_buffer_size;
	sampleBuffers[i] = (char*)aligned_alloc(sizeof(sample_t), sampleBuffersCapacity[i]);
    }

    if (not null) {
	open_samples(dotfile, orig_path, zlevel, &outfile, &outbuf);
    }

    if (nfft) {
	std::cout << "using FFT point size " << nfft << std::endl;

	if (samps_per_buff % nfft) {
	    throw std::runtime_error("FFT point size must be a factor of spb");
	}

	if (rate % nfft_div) {
	    throw std::runtime_error("nfft_div must be a factor of sample rate");
	}

	if (not fftnull) {
	    open_samples(fft_dotfile, orig_path, zlevel, &fft_outfile, &fft_outbuf);
	}
    }

    boost::thread_group writer_threads;
    writer_threads.create_thread(write_samples_worker);
    writer_threads.create_thread(fft_in_worker);
    writer_threads.create_thread(fft_out_worker);

    bool overflow_message = true;
    size_t overflows = 0;

    // setup streaming
    uhd::stream_cmd_t stream_cmd((num_requested_samples == 0)
				     ? uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS
				     : uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.num_samps  = size_t(num_requested_samples);
    stream_cmd.stream_now = true;
    stream_cmd.time_spec  = uhd::time_spec_t();

    typedef std::map<size_t, size_t> SizeMap;
    SizeMap mapSizes;
    const auto start_time = std::chrono::steady_clock::now();

    rx_stream->issue_stream_cmd(stream_cmd);

    const auto stop_time =
	start_time + std::chrono::milliseconds(int64_t(1000 * time_requested));
    // Track time and samps between updating the BW summary
    auto last_update                     = start_time;
    unsigned long long last_update_samps = 0;
    size_t write_ptr = 0;

    // Run this loop until either time expired (if a duration was given), until
    // the requested number of samples were collected (if such a number was
    // given), or until Ctrl-C was pressed.
    for (; not stop_signal_called
	   and (num_requested_samples != num_total_samps or num_requested_samples == 0)
	   and (time_requested == 0.0 or std::chrono::steady_clock::now() < stop_time);) {
	const auto now = std::chrono::steady_clock::now();
	char *buffer_p = sampleBuffers[write_ptr];
	size_t num_rx_samps =
	    rx_stream->recv(buffer_p, max_samples, md, 3.0, enable_size_map);

	if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
	    std::cout << boost::format("Timeout while streaming") << std::endl;
	    break;
	}
	if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
	    ++overflows;
	    if (overflow_message) {
		overflow_message = false;
		std::cerr
		    << boost::format(
			   "Got an overflow indication. Please consider the following:\n"
			   "  Your write medium must sustain a rate of %fMB/s.\n"
			   "  Dropped samples will not be written to the file.\n"
			   "  Please modify this example for your purposes.\n"
			   "  This message will not appear again.\n")
			   % (usrp->get_rx_rate(channel) * sizeof(samp_type) / 1e6);
	    }
	    continue;
	}
	if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
	    std::string error = str(boost::format("Receiver error: %s") % md.strerror());
	    if (continue_on_bad_packet) {
		std::cerr << error << std::endl;
		continue;
	    } else
		throw std::runtime_error(error);
	}

	num_total_samps += num_rx_samps;
	size_t samp_bytes = num_rx_samps * sizeof(samp_type);
	buffer_p = sampleBuffers[write_ptr];
	size_t buffer_capacity = sampleBuffersCapacity[write_ptr];
	if (samp_bytes != buffer_capacity) {
	    std::cout << "resize to " << samp_bytes << " from " << buffer_capacity << std::endl;
	    sampleBuffersCapacity[write_ptr] = samp_bytes;
	}
	if (!sample_queue.push(write_ptr)) {
	    std::cout << "sample buffer queue failed (overflow)" << std::endl;
	}
	if (++write_ptr == kSampleBuffers) {
	    write_ptr = 0;
	}

	if (enable_size_map) {
	    SizeMap::iterator it = mapSizes.find(num_rx_samps);
	    if (it == mapSizes.end())
		mapSizes[num_rx_samps] = 0;
	    mapSizes[num_rx_samps] += 1;
	}

	if (bw_summary) {
	    last_update_samps += num_rx_samps;
	    const auto time_since_last_update = now - last_update;
	    if (time_since_last_update > std::chrono::seconds(1)) {
		const double time_since_last_update_s =
		    std::chrono::duration<double>(time_since_last_update).count();
		const double time_rate = (double)last_update_samps / time_since_last_update_s / 1e6;
		std::cout << "\t" << time_rate << " Msps" << std::endl;
		last_update_samps = 0;
		last_update       = now;
	    }
	}
    }
    samples_input_done = true;
    const auto actual_stop_time = std::chrono::steady_clock::now();
    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    rx_stream->issue_stream_cmd(stream_cmd);
    std::cout << "stream stopped" << std::endl;

    writer_threads.join_all();

    if (!outbuf.empty()) {
	close_samples(file, dotfile, dirname, overflows, &outfile, &outbuf);
    }

    if (!fft_outbuf.empty()) {
	close_samples(fft_file, fft_dotfile, dirname, overflows, &fft_outfile, &fft_outbuf);
    }

    std::cout << "closed" << std::endl;

    for (size_t i = 0; i < kSampleBuffers; ++i) {
	free(sampleBuffers[i]);
    }

    if (stats) {
	std::cout << std::endl;
	const double actual_duration_seconds =
	    std::chrono::duration<float>(actual_stop_time - start_time).count();

	std::cout << boost::format("Received %d samples in %f seconds") % num_total_samps
			 % actual_duration_seconds
		  << std::endl;
	const double time_rate = (double)num_total_samps / actual_duration_seconds / 1e6;
	std::cout << time_rate << " Msps" << std::endl;

	if (enable_size_map) {
	    std::cout << std::endl;
	    std::cout << "Packet size map (bytes: count)" << std::endl;
	    for (SizeMap::iterator it = mapSizes.begin(); it != mapSizes.end(); it++)
		std::cout << it->first << ":\t" << it->second << std::endl;
	}
    }
}

typedef std::function<uhd::sensor_value_t(const std::string&)> get_sensor_fn_t;

bool check_locked_sensor(std::vector<std::string> sensor_names,
    const char* sensor_name,
    get_sensor_fn_t get_sensor_fn,
    double setup_time)
{
    if (std::find(sensor_names.begin(), sensor_names.end(), sensor_name)
	== sensor_names.end())
	return false;

    auto setup_timeout = std::chrono::steady_clock::now()
			 + std::chrono::milliseconds(int64_t(setup_time * 1000));
    bool lock_detected = false;

    std::cout << boost::format("Waiting for \"%s\": ") % sensor_name;
    std::cout.flush();

    while (true) {
	if (lock_detected and (std::chrono::steady_clock::now() > setup_timeout)) {
	    std::cout << " locked." << std::endl;
	    break;
	}
	if (get_sensor_fn(sensor_name).to_bool()) {
	    std::cout << "+";
	    std::cout.flush();
	    lock_detected = true;
	} else {
	    if (std::chrono::steady_clock::now() > setup_timeout) {
		std::cout << std::endl;
		throw std::runtime_error(
		    str(boost::format(
			    "timed out waiting for consecutive locks on sensor \"%s\"")
			% sensor_name));
	    }
	    std::cout << "_";
	    std::cout.flush();
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << std::endl;
    return true;
}

int UHD_SAFE_MAIN(int argc, char* argv[])
{
    // variables to be set by po
    std::string args, file, type, ant, subdev, ref, wirefmt;
    size_t channel, total_num_samps, spb, zlevel, batches, sample_id;
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
	("time", po::value<double>(&total_time), "(DEPRECATED) will go away soon! Use --duration instead")
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
	("wirefmt", po::value<std::string>(&wirefmt)->default_value("sc16"), "wire format (sc8, sc16 or s16)")
	("setup", po::value<double>(&setup_time)->default_value(1.0), "seconds of setup time")
	("progress", "periodically display short-term bandwidth")
	("stats", "show average bandwidth on exit")
	("sizemap", "track packet size and display breakdown on exit")
	("null", "run without writing to file")
	("fftnull", "run without writing to FFT file")
	("continue", "don't abort on a bad packet")
	("skip-lo", "skip checking LO lock status")
	("int-n", "tune USRP with integer-N tuning")
	("nfft", po::value<size_t>(&nfft)->default_value(0), "if > 0, calculate n FFT points")
	("nfft_overlap", po::value<size_t>(&nfft_overlap)->default_value(0), "FFT overlap")
	("nfft_div", po::value<size_t>(&nfft_div)->default_value(50), "calculate FFT over sample rate / n samples (e.g 50 == 20ms)")
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
	std::cout << boost::format("UHD RX samples to file %s") % desc << std::endl;
	std::cout << std::endl
		  << "This application streams data from a single channel of a USRP "
		     "device to a file.\n"
		  << std::endl;
	return ~0;
    }

    bool bw_summary             = vm.count("progress") > 0;
    bool stats                  = vm.count("stats") > 0;
    bool null                   = vm.count("null") > 0;
    bool fftnull                = vm.count("fftnull") > 0;
    bool enable_size_map        = vm.count("sizemap") > 0;
    bool continue_on_bad_packet = vm.count("continue") > 0;
    useVkFFT                    = vm.count("novkfft") == 0;

    if (enable_size_map)
	std::cout << "Packet size tracking enabled - will only recv one packet at a time!"
		  << std::endl;

    if (nfft && type != "short") {
	std::cout << "FFT mode only supported when sample type short" << std::endl;
	return -0;
    }

    if (nfft) {
	if (useVkFFT) {
	    if (init_vkfft(batches, sample_id)) {
		std::cout << "init_vkfft() failed" << std::endl;
		return -0;
	    }
	}
	hammingWindow = arma::conv_to<arma::fvec>::from(sp::hamming(nfft));
	hammingWindowSum = sum(hammingWindow);
    }

    // create a usrp device
    std::cout << std::endl;
    std::cout << boost::format("Creating the usrp device with: %s...") % args
	      << std::endl;
    uhd::usrp::multi_usrp::sptr usrp = uhd::usrp::multi_usrp::make(args);

    // Lock mboard clocks
    if (vm.count("ref")) {
	usrp->set_clock_source(ref);
    }

    // always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("subdev"))
	usrp->set_rx_subdev_spec(subdev);

    std::cout << boost::format("Using Device: %s") % usrp->get_pp_string() << std::endl;

    // set the sample rate
    if (option_rate <= 0.0) {
	std::cerr << "Please specify a valid sample rate" << std::endl;
	return ~0;
    }

    rate = size_t(option_rate);
    std::cout << boost::format("Setting RX Rate: %f Msps...") % (rate / 1e6) << std::endl;
    usrp->set_rx_rate(rate, channel);
    std::cout << boost::format("Actual RX Rate: %f Msps...")
		     % (usrp->get_rx_rate(channel) / 1e6)
	      << std::endl
	      << std::endl;

    // set the center frequency
    if (vm.count("freq")) { // with default of 0.0 this will always be true
	std::cout << boost::format("Setting RX Freq: %f MHz...") % (freq / 1e6)
		  << std::endl;
	std::cout << boost::format("Setting RX LO Offset: %f MHz...") % (lo_offset / 1e6)
		  << std::endl;
	uhd::tune_request_t tune_request(freq, lo_offset);
	if (vm.count("int-n"))
	    tune_request.args = uhd::device_addr_t("mode_n=integer");
	usrp->set_rx_freq(tune_request, channel);
	std::cout << boost::format("Actual RX Freq: %f MHz...")
			 % (usrp->get_rx_freq(channel) / 1e6)
		  << std::endl
		  << std::endl;
    }

    // set the rf gain
    if (vm.count("gain")) {
	std::cout << boost::format("Setting RX Gain: %f dB...") % gain << std::endl;
	usrp->set_rx_gain(gain, channel);
	std::cout << boost::format("Actual RX Gain: %f dB...")
			 % usrp->get_rx_gain(channel)
		  << std::endl
		  << std::endl;
    }

    // set the IF filter bandwidth
    if (vm.count("bw")) {
	std::cout << boost::format("Setting RX Bandwidth: %f MHz...") % (bw / 1e6)
		  << std::endl;
	usrp->set_rx_bandwidth(bw, channel);
	std::cout << boost::format("Actual RX Bandwidth: %f MHz...")
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
	std::cout << "Press Ctrl + C to stop streaming..." << std::endl;
    }

#define recv_to_file_args(format) \
    (usrp,                        \
	format,                   \
	wirefmt,                  \
	channel,                  \
	file,                     \
	spb,                      \
	zlevel,                   \
	total_num_samps,          \
	total_time,               \
	bw_summary,               \
	stats,                    \
	null,                     \
	fftnull,                  \
	enable_size_map,          \
	continue_on_bad_packet)
    // recv to file
    if (wirefmt == "s16") {
	if (type == "double")
	    recv_to_file<double> recv_to_file_args("f64");
	else if (type == "float")
	    recv_to_file<float> recv_to_file_args("f32");
	else if (type == "short")
	    recv_to_file<short> recv_to_file_args("s16");
	else
	    throw std::runtime_error("Unknown type " + type);
    } else {
	if (type == "double")
	    recv_to_file<std::complex<double>> recv_to_file_args("fc64");
	else if (type == "float")
	    recv_to_file<std::complex<float>> recv_to_file_args("fc32");
	else if (type == "short")
	    recv_to_file<std::complex<short>> recv_to_file_args("sc16");
	else
	    throw std::runtime_error("Unknown type " + type);
    }

    if (nfft) {
       if (useVkFFT) {
	   free_vkfft();
       }
       if (ffts_in != ffts_out) {
	   std::cout << "fft in/out mismatch - FFT pipeline overflow? increase kFFTbufferCount" << std::endl;
       }
    }

    // finished
    std::cout << std::endl << "Done!" << std::endl << std::endl;

    return EXIT_SUCCESS;
}
