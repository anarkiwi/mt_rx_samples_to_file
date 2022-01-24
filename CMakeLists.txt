#
# Copyright 2010-2015 Ettus Research LLC
# Copyright 2018 Ettus Research, a National Instruments Company
#
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.16)

project(mt_rx_samples_to_file)
set(CMAKE_BUILD_TYPE Release)
add_executable(mt_rx_samples_to_file mt_rx_samples_to_file.cpp)

find_package(UHD 3.15.0 REQUIRED)
find_package(Boost ${Boost_Version} COMPONENTS iostreams thread program_options filesystem REQUIRED)
target_link_libraries(mt_rx_samples_to_file LINK_PUBLIC ${UHD_LIBRARIES} ${Boost_LIBRARIES})
