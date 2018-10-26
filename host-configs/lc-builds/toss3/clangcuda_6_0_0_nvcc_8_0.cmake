##############################################################################
## Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

set(COMB_COMPILER "COMB_COMPILER_CLANG" CACHE STRING "")

set(CUDA_SEPERABLE_COMPILATION ON CACHE BOOL "")
set(CUDA_LINK_WITH_NVCC ON CACHE BOOL "")

set(MPI_CXX_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-6.0.0/bin/mpic++" CACHE PATH "")
set(MPI_C_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-6.0.0/bin/mpicc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-6.0.0/bin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-6.0.0/bin/clang" CACHE PATH "")

set(CMAKE_EXE_LINKER_FLAGS "-L/usr/tce/packages/cuda/cuda-8.0/lib64 -lcudart_static -lcudadevrt -lrt -ldl -lnvToolsExt -pthread -Wl,-rpath=/usr/tce/packages/clang/clang-6.0.0/lib" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.2 -funroll-loops -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -msse4.2 -funroll-loops -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(COMB_HOST_CONFIG_LOADED On CACHE Bool "")
