#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "sample_pipeline.h"

#include "sigpack/sigpack.h"


BOOST_AUTO_TEST_CASE(SmokeTest)
{
    init_sample_buffers(1e3*1024, 4);
    sample_pipeline_start("short", "", "", 1, false, 0, 0, 1, 1, 1e6, 0, 0);
    sample_pipeline_stop(0);
}


BOOST_AUTO_TEST_CASE(RandomFFTTest)
{
    using namespace boost::filesystem;
    path tmpdir = temp_directory_path() / unique_path();
    create_directory(tmpdir);
    std::string file = tmpdir.string() + "/samples.zst";
    std::string fft_file = tmpdir.string() + "/fft_samples.zst";
    arma::arma_rng::set_seed_random();
    arma::Col<std::complex<float>> samples(1e3*1024);
    samples.randu();
    init_sample_buffers(samples.size() * sizeof(std::complex<float>), sizeof(std::complex<float>));
    sample_pipeline_start("float", file, file, 1, true, 256, 128, 1, 1, samples.size(), 100, 0);
    size_t buffer_capacity;
    size_t write_ptr = 0;
    char *buffer_p = get_sample_buffer(write_ptr, &buffer_capacity);
    memcpy(buffer_p, samples.memptr(), samples.size() * sizeof(std::complex<float>));
    enqueue_samples(write_ptr);
    sample_pipeline_stop(0);
    remove_all(tmpdir);
}

