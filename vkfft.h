void free_vkfft();
int64_t init_vkfft(size_t batches, size_t sample_id, size_t nfft);
void vkfft_specgram_offload(arma::cx_fmat &Pw_in, arma::cx_fmat &Pw);
