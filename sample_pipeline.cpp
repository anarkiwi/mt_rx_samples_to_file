#include "sample_pipeline.h"

const size_t kSampleBuffers = 8;
const size_t kFFTbuffers = 256;

static arma::fvec hammingWindow;
static float hammingWindowSum = 0;

static std::pair<arma::cx_fmat, arma::cx_fmat> FFTBuffers[kFFTbuffers];
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kFFTbuffers>> in_fft_queue;
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kFFTbuffers>> out_fft_queue;
static std::pair<boost::scoped_ptr<char>, size_t> sampleBuffers[kSampleBuffers];
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kSampleBuffers>> sample_queue;
static arma::cx_fvec fft_samples_in;


void enqueue_samples(size_t &buffer_ptr) {
    if (!sample_queue.push(buffer_ptr)) {
        std::cout << "sample buffer queue failed (overflow)" << std::endl;
        return;
    }

    if (++buffer_ptr == kSampleBuffers) {
        buffer_ptr = 0;
    }
}


void set_sample_buffer_capacity(size_t buffer_ptr, size_t buffer_size) {
    sampleBuffers[buffer_ptr].second = buffer_size;
}


void init_sample_buffers(size_t max_buffer_size, size_t samp_size) {
    for (size_t i = 0; i < kSampleBuffers; ++i) {
        set_sample_buffer_capacity(i, max_buffer_size);
        sampleBuffers[i].first.reset((char*)aligned_alloc(samp_size, max_buffer_size));
    }
}


char *get_sample_buffer(size_t buffer_ptr, size_t *buffer_capacity) {
    if (buffer_capacity) {
        *buffer_capacity = sampleBuffers[buffer_ptr].second;
    }
    return sampleBuffers[buffer_ptr].first.get();
}


bool dequeue_samples(size_t &read_ptr) {
    return sample_queue.pop(read_ptr);
}



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


void specgram_window(arma::cx_fmat &Pw_in, const arma::uword Nfft, const arma::uword Noverl)
{
    arma::uword N = fft_samples_in.size();
    arma::uword D = Nfft-Noverl;
    arma::uword m = 0;
    const arma::uword U = static_cast<arma::uword>(floor((N-Noverl)/double(D)));
    Pw_in.set_size(Nfft,U);

    for(arma::uword k=0; k<=N-Nfft; k+=D)
    {
	Pw_in.col(m++) = fft_samples_in.rows(k,k+Nfft-1) % hammingWindow;
    }
}


void queue_fft(size_t &fft_write_ptr, size_t nfft, size_t nfft_overlap) {
    arma::cx_fmat &Pw_in = FFTBuffers[fft_write_ptr].first;
    specgram_window(Pw_in, nfft, nfft_overlap);
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


template <typename samp_type>
void write_samples(SampleWriter *sample_writer, size_t &fft_write_ptr, size_t &curr_nfft_ds, size_t nfft, size_t nfft_overlap, size_t nfft_ds)
{
    size_t read_ptr;
    size_t buffer_capacity = 0;
    while (dequeue_samples(read_ptr)) {
        char *buffer_p = get_sample_buffer(read_ptr, &buffer_capacity);
        if (nfft) {
            samp_type *i_p = (samp_type*)buffer_p;
            for (size_t i = 0; i < buffer_capacity / (fft_samples_in.size() * sizeof(samp_type)); ++i) {
                for (size_t fft_p = 0; fft_p < fft_samples_in.size(); ++fft_p, ++i_p) {
                    fft_samples_in[fft_p] = std::complex<float>(i_p->real(), i_p->imag());
                }
                if (++curr_nfft_ds == nfft_ds) {
                    curr_nfft_ds = 0;
                    queue_fft(fft_write_ptr, nfft, nfft_overlap);
                }
            }
        }
        sample_writer->write(buffer_p, buffer_capacity);
        std::cout << "." << std::endl;
    }
}


void wrap_write_samples(const std::string &type, SampleWriter *sample_writer, size_t &fft_write_ptr, size_t &curr_nfft_ds, size_t nfft, size_t nfft_overlap, size_t nfft_ds)
{
    if (type == "double")
        write_samples<std::complex<double>>(sample_writer, fft_write_ptr, curr_nfft_ds, nfft, nfft_overlap, nfft_ds);
    else if (type == "float")
        write_samples<std::complex<float>>(sample_writer, fft_write_ptr, curr_nfft_ds, nfft, nfft_overlap, nfft_ds);
    else if (type == "short")
        write_samples<std::complex<short>>(sample_writer, fft_write_ptr, curr_nfft_ds, nfft, nfft_overlap, nfft_ds);
    else
        throw std::runtime_error("Unknown type " + type);
}


void write_samples_worker(const std::string &type, SampleWriter *sample_writer, boost::atomic<bool> *samples_input_done, boost::atomic<bool> *write_samples_worker_done, size_t nfft, size_t nfft_overlap, size_t nfft_div, size_t nfft_ds, size_t rate)
{
    size_t fft_write_ptr = 0;
    size_t curr_nfft_ds = 0;
    fft_samples_in.set_size(rate / nfft_div);

    while (!*samples_input_done) {
        wrap_write_samples(type, sample_writer, fft_write_ptr, curr_nfft_ds, nfft, nfft_overlap, nfft_ds);
        usleep(10000);
    }

    wrap_write_samples(type, sample_writer, fft_write_ptr, curr_nfft_ds, nfft, nfft_overlap, nfft_ds);
    *write_samples_worker_done = true;
    std::cout << "write samples worker done" << std::endl;
}


void sample_pipeline_start() {

}


void sample_pipeline_stop() {

}
