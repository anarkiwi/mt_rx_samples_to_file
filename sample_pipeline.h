#include "sigpack/sigpack.h"
#include "sample_writer.h"

void specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw);
void fft_out_offload(SampleWriter *fft_sample_writer, arma::cx_fmat &Pw, float hammingWindowSum);
