#
# Copyright 2010-2015 Ettus Research LLC
# Copyright 2018 Ettus Research, a National Instruments Company
#
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.16)

project(mt_rx_samples_to_file)
set(CMAKE_BUILD_TYPE Release)
include(FindPkgConfig)
include_directories(VkFFT/glslang-master/glslang/Include VkFFT/vkFFT VkFFT/benchmark_scripts/vkFFT_scripts/include VkFFT/glslang-master/glslang/Public)

find_package(Vulkan REQUIRED)
find_package(Armadillo REQUIRED)
find_package(UHD 3.15.0 REQUIRED)
find_package(Boost ${Boost_Version} COMPONENTS iostreams thread program_options filesystem REQUIRED)

add_executable(mt_rx_samples_to_file mt_rx_samples_to_file.cpp sample_writer.cpp vkfft.cpp)
target_link_libraries(mt_rx_samples_to_file LINK_PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/glslang/libglslang.a ${ARMADILLO_LIBRARIES} ${UHD_LIBRARIES} ${Boost_LIBRARIES} ${Vulkan_LIBRARIES} ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/glslang/libMachineIndependent.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/glslang/OSDependent/Unix/libOSDependent.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/OGLCompilersDLL/libOGLCompiler.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/glslang/libGenericCodeGen.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/glslang-master/SPIRV/libSPIRV.a ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/build/CMakeFiles/Vulkan_FFT.dir/benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp.o)
