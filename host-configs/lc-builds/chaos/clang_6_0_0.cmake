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

set(MPI_CXX_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-6.0.0/bin/mpiclang++" CACHE PATH "")
set(MPI_C_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-6.0.0/bin/mpiclang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-6.0.0/bin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-6.0.0/bin/clang" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.2 -funroll-loops -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -msse4.2 -funroll-loops -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(COMB_HOST_CONFIG_LOADED On CACHE BOOL "")
