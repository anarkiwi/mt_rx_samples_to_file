#include "sample_pipeline.h"


void specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw) {
    const size_t nfft_rows = Pw_in.n_rows;

    for(arma::uword k=0; k < Pw_in.n_cols; ++k)
    {
        Pw.col(k) = arma::fft(Pw_in.col(k), nfft_rows);
    }
}


void fft_out_offload(SampleWriter *fft_sample_writer, arma::cx_fmat &Pw, float hammingWindowSum) {
    Pw /= hammingWindowSum;
    // TODO: offload C2R
    arma::fmat fft_points_out = log10(real(Pw % conj(Pw))) * 10;
    fft_sample_writer->write((const char*)fft_points_out.memptr(), fft_points_out.n_elem * sizeof(float));
}
