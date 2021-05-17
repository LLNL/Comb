##############################################################################
## Copyright (c) 2018-2021, Lawrence Livermore National Security, LLC.
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

set(COMB_COMPILER "COMB_COMPILER_XLC" CACHE STRING "")

set(MPI_CXX_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-xl-2019.02.07/bin/mpixlC" CACHE PATH "")
set(MPI_C_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-xl-2019.02.07/bin/mpixlc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/xl/xl-2019.02.07/bin/xlc++_r" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/xl/xl-2019.02.07/bin/xlC_r" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 " CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g " CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -qsmp=omp:noopt " CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,muldefs" CACHE STRING "")

set(COMB_HOST_CONFIG_LOADED On CACHE BOOL "")
