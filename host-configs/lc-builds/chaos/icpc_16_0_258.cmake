##############################################################################
## Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

set(COMB_COMPILER "COMB_COMPILER_ICC" CACHE STRING "")

set(MPI_CXX_COMPILER "/usr/local/bin/mpiicpc-16.0.258" CACHE PATH "")
set(MPI_C_COMPILER "/usr/local/bin/mpiicc-16.0.258" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc-16.0.258" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/local/bin/icc-16.0.258" CACHE PATH "")

set(COMMON_FLAGS "-gnu-prefix=/usr/apps/gnu/4.9.3/bin/ -Wl,-rpath,/usr/apps/gnu/4.9.3/lib64 -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -march=native -ansi-alias" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g -march=native -ansi-alias" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -O0 -g" CACHE STRING "")

set(COMB_HOST_CONFIG_LOADED On CACHE Bool "")
