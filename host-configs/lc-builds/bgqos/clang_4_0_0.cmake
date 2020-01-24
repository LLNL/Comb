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

set(COMB_COMPILER "COMB_COMPILER_CLANG" CACHE STRING "")

set(MPI_CXX_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/mpiclang++11" CACHE PATH "")
set(MPI_C_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/mpiclang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/bgclang++11" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/bgclang" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -std=c++11 -stdlib=libc++" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ffast-math -std=c++11 -stdlib=libc++" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -std=c++11 -stdlib=libc++" CACHE STRING "")

set(MPIEXEC              "/usr/bin/srun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-n" CACHE PATH "")

set(ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC TRUE CACHE BOOL "Ensures that tests will be wrapped with srun to run on the backend nodes")

set(COMB_HOST_CONFIG_LOADED On CACHE Bool "")
