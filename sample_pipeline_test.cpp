#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "sample_pipeline.h"


BOOST_AUTO_TEST_CASE(SmokeTest)
{
    init_sample_buffers(1e3*1024, 4);
    sample_pipeline_start("short", "", "", 1, false, 0, 0, 1, 1, 1e6, 0, 0);
    sample_pipeline_stop(0);
}
