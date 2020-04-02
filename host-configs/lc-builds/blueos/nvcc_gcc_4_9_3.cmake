##############################################################################
## Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory
##
## LLNL-CODE-758885
##
## All rights reserved.
##
## This file is part of Comb.
##
## For details, see https://github.com/LLNL/Comb
## Please also see the LICENSE file for MIT license.
##############################################################################

set(COMB_COMPILER "COMB_COMPILER_GNU" CACHE STRING "")

set(MPI_CXX_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-4.9.3/bin/mpig++" CACHE PATH "")
set(MPI_C_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-4.9.3/bin/mpigcc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-4.9.3/bin/g++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-4.9.3/bin/gcc" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE        "-Ofast -finline-functions"    CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -g -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os"                          CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g"                       CACHE STRING "")


if(NOT DEFINED CUDA_ARCH)
  message(FATAL_ERROR "CUDA_ARCH NOT DEFINED")
endif()

set(COMB_NVCC_FLAGS "-restrict -arch ${CUDA_ARCH} --expt-extended-lambda" CACHE STRING "")
set(COMB_NVCC_FLAGS_RELEASE        "-O3"                                             CACHE STRING "")
set(COMB_NVCC_FLAGS_RELWITHDEBINFO "-O2 -g -lineinfo"                                CACHE STRING "")
set(COMB_NVCC_FLAGS_MINSIZEREL     "-Os"                                             CACHE STRING "")
set(COMB_NVCC_FLAGS_DEBUG          "-O0 -g -G"                                       CACHE STRING "")

set(COMB_HOST_CONFIG_LOADED On CACHE Bool "")
