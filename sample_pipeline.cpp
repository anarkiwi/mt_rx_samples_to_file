#include "sample_pipeline.h"


static arma::fvec hammingWindow;
static float hammingWindowSum = 0;


void init_hamming_window(size_t nfft) {
    hammingWindow = arma::conv_to<arma::fvec>::from(sp::hamming(nfft));
    hammingWindowSum = sum(hammingWindow);
}


void specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw) {
    const size_t nfft_rows = Pw_in.n_rows;

    for(arma::uword k=0; k < Pw_in.n_cols; ++k)
    {
        Pw.col(k) = arma::fft(Pw_in.col(k), nfft_rows);
    }
}


void fft_out_offload(SampleWriter *fft_sample_writer, arma::cx_fmat &Pw) {
    Pw /= hammingWindowSum;
    // TODO: offload C2R
    arma::fmat fft_points_out = log10(real(Pw % conj(Pw))) * 10;
    fft_sample_writer->write((const char*)fft_points_out.memptr(), fft_points_out.n_elem * sizeof(float));
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
