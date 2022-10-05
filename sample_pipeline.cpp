#include "sample_pipeline.h"

const size_t kFFTbuffers = 256;

static arma::fvec hammingWindow;
static float hammingWindowSum = 0;

static std::pair<arma::cx_fmat, arma::cx_fmat> FFTBuffers[kFFTbuffers];
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kFFTbuffers>> in_fft_queue;
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kFFTbuffers>> out_fft_queue;


typedef void (*offload_p)(arma::cx_fmat&, arma::cx_fmat&);


void specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw) {
    const size_t nfft_rows = Pw_in.n_rows;

    for(arma::uword k=0; k < Pw_in.n_cols; ++k)
    {
	Pw.col(k) = arma::fft(Pw_in.col(k), nfft_rows);
    }
}


inline void fftin(offload_p offload) {
    size_t read_ptr;
    while (in_fft_queue.pop(read_ptr)) {
	offload(FFTBuffers[read_ptr].first, FFTBuffers[read_ptr].second);
	while (!out_fft_queue.push(read_ptr)) {
	    usleep(100);
	}
    }
}


void fft_in_worker(bool useVkFFT, boost::atomic<bool> *write_samples_worker_done, boost::atomic<bool> *fft_in_worker_done)
{
    offload_p offload = specgram_offload;
    if (useVkFFT) {
	offload = vkfft_specgram_offload;
    }
    while (!*write_samples_worker_done) {
	fftin(offload);
	usleep(10000);
    }
    fftin(offload);
    *fft_in_worker_done = true;
    std::cout << "fft worker done" << std::endl;
}


void fftout(SampleWriter *fft_sample_writer) {
    size_t read_ptr;
    while (out_fft_queue.pop(read_ptr)) {
	fft_out_offload(fft_sample_writer, FFTBuffers[read_ptr].second);
    }
}


void fft_out_worker(SampleWriter *fft_sample_writer, boost::atomic<bool> *fft_in_worker_done)
{
    while (!*fft_in_worker_done) {
	fftout(fft_sample_writer);
	usleep(10000);
    }
    fftout(fft_sample_writer);
    std::cout << "fft out worker done" << std::endl;
}


void specgram_window(const arma::cx_fvec& x, arma::cx_fmat &Pw_in, const arma::uword Nfft, const arma::uword Noverl)
{
    arma::uword N = x.size();
    arma::uword D = Nfft-Noverl;
    arma::uword m = 0;
    const arma::uword U = static_cast<arma::uword>(floor((N-Noverl)/double(D)));
    Pw_in.set_size(Nfft,U);

    for(arma::uword k=0; k<=N-Nfft; k+=D)
    {
	Pw_in.col(m++) = x.rows(k,k+Nfft-1) % hammingWindow;
    }
}


void queue_fft(const arma::cx_fvec &fft_samples_in, size_t &fft_write_ptr, size_t nfft, size_t nfft_overlap) {
    arma::cx_fmat &Pw_in = FFTBuffers[fft_write_ptr].first;
    specgram_window(fft_samples_in, Pw_in, nfft, nfft_overlap);
    arma::cx_fmat &Pw = FFTBuffers[fft_write_ptr].second;
    Pw.copy_size(Pw_in);
    while (!in_fft_queue.push(fft_write_ptr)) {
	usleep(100);
    }
    if (++fft_write_ptr == kFFTbuffers) {
	fft_write_ptr = 0;
    }
}


void init_hamming_window(size_t nfft) {
    hammingWindow = arma::conv_to<arma::fvec>::from(sp::hamming(nfft));
    hammingWindowSum = sum(hammingWindow);
}


void fft_out_offload(SampleWriter *fft_sample_writer, arma::cx_fmat &Pw) {
    Pw /= hammingWindowSum;
    // TODO: offload C2R
    arma::fmat fft_points_out = log10(real(Pw % conj(Pw))) * 10;
    fft_sample_writer->write((const char*)fft_points_out.memptr(), fft_points_out.n_elem * sizeof(float));
}
