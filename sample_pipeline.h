#include <boost/atomic.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/scoped_ptr.hpp>

#include "sigpack/sigpack.h"
#include "sample_writer.h"
#include "vkfft.h"

void init_hamming_window(size_t nfft);
void queue_fft(size_t &fft_write_ptr, size_t nfft, size_t nfft_overlap);
void fft_out_offload(SampleWriter *fft_sample_writer, arma::cx_fmat &Pw);
void fft_out_worker(SampleWriter *fft_sample_writer, boost::atomic<bool> *fft_in_worker_done);
void fft_in_worker(bool useVkFFT, boost::atomic<bool> *write_samples_worker_done, boost::atomic<bool> *fft_in_worker_done);
void enqueue_samples(size_t &buffer_ptr);
void set_sample_buffer_capacity(size_t buffer_ptr, size_t buffer_size);
void init_sample_buffers(size_t max_buffer_size, size_t samp_size);
char *get_sample_buffer(size_t buffer_ptr, size_t *buffer_capacity);
bool dequeue_samples(size_t &read_ptr);
void write_samples_worker(const std::string &type, SampleWriter *sample_writer, boost::atomic<bool> *samples_input_done, boost::atomic<bool> *write_samples_worker_done, size_t nfft, size_t nfft_overlap, size_t nfft_div, size_t nfft_ds, size_t rate);
void sample_pipeline_start();
void sample_pipeline_stop();
