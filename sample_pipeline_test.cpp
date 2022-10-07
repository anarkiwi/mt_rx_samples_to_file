#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "sample_pipeline.h"

#include "sigpack/sigpack.h"


BOOST_AUTO_TEST_CASE(SmokeTest)
{
    sample_pipeline_start("short", "", "", 1e6, 1, false, 0, 0, 1, 1, 1e6, 0, 0);
    sample_pipeline_stop(0);
}


BOOST_AUTO_TEST_CASE(RandomFFTTest)
{
    using namespace boost::filesystem;
    path tmpdir = temp_directory_path() / unique_path();
    create_directory(tmpdir);
    std::string file = tmpdir.string() + "/samples.dat";
    std::string fft_file = tmpdir.string() + "/fft_samples.dat";
    arma::arma_rng::set_seed_random();
    arma::Col<std::complex<float>> samples(1e3 * 1024);
    samples.randu();
    sample_pipeline_start("float", file, fft_file, samples.size(), 1, false, 256, 128, 1, 1, samples.size(), 100, 0);
    size_t buffer_capacity;
    size_t write_ptr = 0;
    char *buffer_p = get_sample_buffer(write_ptr, &buffer_capacity);
    memcpy(buffer_p, samples.memptr(), samples.size() * sizeof(std::complex<float>));
    enqueue_samples(write_ptr);
    sample_pipeline_stop(0);
    arma::Col<std::complex<float>> disk_samples;
    disk_samples.copy_size(samples);
    FILE *samples_fp = fopen(file.c_str(), "rb");
    int samples_bytes = fread(disk_samples.memptr(), sizeof(std::complex<float>), disk_samples.size(), samples_fp);
    fclose(samples_fp);
    BOOST_TEST(samples_bytes == samples.size());
    BOOST_TEST(arma::all(samples == disk_samples));
    FILE *fft_samples_fp = fopen(fft_file.c_str(), "rb");
    arma::fvec fft(samples.size());
    int fft_bytes = fread(fft.memptr(), sizeof(float), fft.size(), fft_samples_fp);
    BOOST_TEST(fft_bytes == fft.size());
    float mean = arma::mean(fft);
    BOOST_TEST(((mean > -20) && (mean < 0)));
    fclose(fft_samples_fp);
    remove_all(tmpdir);
}

