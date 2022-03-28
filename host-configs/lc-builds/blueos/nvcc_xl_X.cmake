##############################################################################
## Copyright (c) 2018-2022, Lawrence Livermore National Security, LLC.
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

set(CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g9" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -qsmp=omp:noopt" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,muldefs" CACHE STRING "")

set(HOST_OPT_FLAGS "-Xcompiler -O3 -Xcompiler -qxlcompatmacros -Xcompiler -qalias=noansi -Xcompiler -qsmp=omp -Xcompiler -qhot -Xcompiler -qnoeh -Xcompiler -qsuppress=1500-029 -Xcompiler -qsuppress=1500-036")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 ${HOST_OPT_FLAGS}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo -O3 ${HOST_OPT_FLAGS}" CACHE STRING "")

# Suppressed XLC warnings:
# - 1500-029 cannot inline
# - 1500-036 nostrict optimizations may alter code semantics
#   (can be countered with -qstrict, with less optimization)

set(COMB_HOST_CONFIG_LOADED On CACHE BOOL "")
