#include "sample_pipeline.h"

void specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw) {
    const size_t nfft_rows = Pw_in.n_rows;

    for(arma::uword k=0; k < Pw_in.n_cols; ++k)
    {
        Pw.col(k) = arma::fft(Pw_in.col(k), nfft_rows);
    }
}
