#
# Copyright 2010-2015 Ettus Research LLC
# Copyright 2018 Ettus Research, a National Instruments Company
#
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.16)

project(mt_rx_samples_to_file)
set(CMAKE_BUILD_TYPE Release)
include(FindPkgConfig)
include(CTest)

find_package(Vulkan REQUIRED)
find_package(Armadillo REQUIRED)
find_package(UHD 3.15.0 REQUIRED)
find_package(Boost ${Boost_Version} COMPONENTS filesystem iostreams thread unit_test_framework program_options REQUIRED)

add_library(vkfft vkfft.cpp)
target_link_libraries(vkfft ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/glslang/libglslang.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/glslang/libMachineIndependent.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/glslang/OSDependent/Unix/libOSDependent.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/OGLCompilersDLL/libOGLCompiler.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/glslang/libGenericCodeGen.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/SPIRV/libSPIRV.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/CMakeFiles/Vulkan_FFT.dir/benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp.o)
target_include_directories(vkfft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/glslang-master/glslang/Include ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/glslang-master/glslang/Public ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/vkFFT ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/benchmark_scripts/vkFFT_scripts/include)

add_library(sample_writer sample_writer.cpp)

add_library(sample_pipeline sample_pipeline.cpp)
target_link_libraries(sample_pipeline vkfft ${ARMADILLO_LIBRARIES} ${Boost_LIBRARIES} ${Vulkan_LIBRARIES})

add_executable(mt_rx_samples_to_file mt_rx_samples_to_file.cpp)
target_link_libraries(mt_rx_samples_to_file sample_writer sample_pipeline ${Boost_LIBRARIES} ${UHD_LIBRARIES})

add_executable(sample_pipeline_test sample_pipeline_test.cpp)
target_link_libraries(sample_pipeline_test ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_test(NAME sample_pipeline_test COMMAND sample_pipeline_test)
