#include <boost/atomic.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include "sigpack/sigpack.h"
#include "sample_writer.h"
#include "vkfft.h"

void init_hamming_window(size_t nfft);
void queue_fft(const arma::cx_fvec &fft_samples_in, size_t &fft_write_ptr, size_t nfft, size_t nfft_overlap);
void fft_out_offload(SampleWriter *fft_sample_writer, arma::cx_fmat &Pw);
void fft_out_worker(SampleWriter *fft_sample_writer, boost::atomic<bool> *fft_in_worker_done);
void fft_in_worker(bool useVkFFT, boost::atomic<bool> *write_samples_worker_done, boost::atomic<bool> *fft_in_worker_done);
