#include <boost/atomic.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread/thread.hpp>

#include "sigpack/sigpack.h"

#include "sample_pipeline.h"
#include "sample_writer.h"
#include "vkfft.h"

typedef void (*offload_p)(arma::cx_fmat&, arma::cx_fmat&);

const size_t kSampleBuffers = 8;
const size_t kFFTbuffers = 256;

static arma::fvec hammingWindow;
static float hammingWindowSum = 0;

static size_t nfft = 0, nfft_overlap = 0,  nfft_ds = 0, samp_size = 0;
static bool useVkFFT = false;
static offload_p offload;
static void(*write_samples_p)(size_t &, size_t &);

static std::pair<arma::cx_fmat, arma::cx_fmat> FFTBuffers[kFFTbuffers];
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kFFTbuffers>> in_fft_queue;
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kFFTbuffers>> out_fft_queue;
static std::pair<boost::scoped_ptr<char>, size_t> sampleBuffers[kSampleBuffers];
boost::lockfree::spsc_queue<size_t, boost::lockfree::capacity<kSampleBuffers>> sample_queue;
static arma::cx_fvec fft_samples_in;
static boost::atomic<bool> samples_input_done(false);
static boost::atomic<bool> write_samples_worker_done(false);
static boost::atomic<bool> fft_in_worker_done(false);
boost::scoped_ptr<SampleWriter> sample_writer;
boost::scoped_ptr<SampleWriter> fft_sample_writer;
boost::thread_group writer_threads;


void enqueue_samples(size_t &buffer_ptr) {
    if (!sample_queue.push(buffer_ptr)) {
	std::cerr << "sample buffer queue failed (overflow)" << std::endl;
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


void specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw) {
    const size_t nfft_rows = Pw_in.n_rows;

    for(arma::uword k=0; k < Pw_in.n_cols; ++k)
    {
	Pw.col(k) = arma::fft(Pw_in.col(k), nfft_rows);
    }
}


inline void fftin() {
    size_t read_ptr;
    while (in_fft_queue.pop(read_ptr)) {
	offload(FFTBuffers[read_ptr].first, FFTBuffers[read_ptr].second);
	while (!out_fft_queue.push(read_ptr)) {
	    usleep(100);
	}
    }
}


void fft_in_worker()
{
    while (!write_samples_worker_done) {
	fftin();
	usleep(10000);
    }
    fftin();
    fft_in_worker_done = true;
    std::cerr << "fft worker done" << std::endl;
}


void fft_out_offload(const arma::cx_fmat &Pw) {
    // TODO: offload C2R
    arma::fmat fft_points_out = log10(real(Pw % conj(Pw / hammingWindowSum))) * 10;
    fft_sample_writer->write((const char*)fft_points_out.memptr(), fft_points_out.n_elem * sizeof(float));
}


void fftout() {
    size_t read_ptr;
    while (out_fft_queue.pop(read_ptr)) {
	fft_out_offload(FFTBuffers[read_ptr].second);
    }
}


void fft_out_worker()
{
    while (!fft_in_worker_done) {
	fftout();
	usleep(10000);
    }
    fftout();
    std::cerr << "fft out worker done" << std::endl;
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


void queue_fft(size_t &fft_write_ptr) {
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


template <typename samp_type>
void write_samples(size_t &fft_write_ptr, size_t &curr_nfft_ds)
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
		    queue_fft(fft_write_ptr);
		}
	    }
	}
	sample_writer->write(buffer_p, buffer_capacity);
	std::cerr << "." << std::endl;
    }
}


void write_samples_worker()
{
    size_t fft_write_ptr = 0;
    size_t curr_nfft_ds = 0;

    while (!samples_input_done) {
	write_samples_p(fft_write_ptr, curr_nfft_ds);
	usleep(10000);
    }

    write_samples_p(fft_write_ptr, curr_nfft_ds);
    write_samples_worker_done = true;
    std::cerr << "write samples worker done" << std::endl;
}

size_t get_samp_size() {
    return samp_size;
}


void sample_pipeline_start(const std::string &type, const std::string &file, const std::string &fft_file, size_t zlevel, bool useVkFFT_, size_t nfft_, size_t nfft_overlap_, size_t nfft_div, size_t nfft_ds_, size_t rate, size_t batches, size_t sample_id) {
    nfft = nfft_;
    nfft_overlap_ = nfft_overlap_;
    nfft_ds = nfft_ds_;
    useVkFFT = useVkFFT_;

    offload = specgram_offload;
    if (useVkFFT) {
	offload = vkfft_specgram_offload;
	init_vkfft(batches, sample_id, nfft);
    }
    if (type == "double") {
        write_samples_p = &write_samples<std::complex<double>>;
        samp_size = sizeof(std::complex<double>);
    } else if (type == "float") { 
        write_samples_p = &write_samples<std::complex<float>>;
        samp_size = sizeof(std::complex<float>);
    } else if (type == "short") {
        write_samples_p = &write_samples<std::complex<short>>;
        samp_size = sizeof(std::complex<short>);
    } else {
        throw std::runtime_error("Unknown type " + type);
    }
    init_hamming_window(nfft);
    fft_samples_in.set_size(rate / nfft_div);
    samples_input_done = false;
    write_samples_worker_done = false;
    fft_in_worker_done = false;
    sample_writer.reset(new SampleWriter());
    fft_sample_writer.reset(new SampleWriter());
    if (file.size()) {
	sample_writer->open(file, zlevel);
    }
    if (fft_file.size()) {
	fft_sample_writer->open(fft_file, zlevel);
    }
    writer_threads.add_thread(new boost::thread(write_samples_worker));
    writer_threads.add_thread(new boost::thread(fft_in_worker));
    writer_threads.add_thread(new boost::thread(fft_out_worker));
}


void sample_pipeline_stop(size_t overflows) {
    samples_input_done = true;
    writer_threads.join_all();
    sample_writer->close(overflows);
    fft_sample_writer->close(overflows);
    if (useVkFFT) {
	free_vkfft();
    }
}
