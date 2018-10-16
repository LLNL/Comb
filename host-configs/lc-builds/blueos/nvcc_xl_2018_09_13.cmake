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

set(COMB_COMPILER "COMB_COMPILER_XLC" CACHE STRING "")

set(MPI_CXX_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-xl-beta-2018.09.13/bin/mpixlC" CACHE PATH "")
set(MPI_C_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-xl-beta-2018.09.13/bin/mpixlc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/xl/xl-beta-2018.09.13/bin/xlc++_r" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/xl/xl-beta-2018.09.13/bin/xlC_r" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3  " CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g9 " CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -qsmp=omp:noopt " CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,muldefs" CACHE STRING "")

set(CUDA_COMMON_OPT_FLAGS -restrict; -arch ${CUDA_ARCH}; -std c++11;  --generate-line-info; --expt-extended-lambda; --expt-relaxed-constexpr)
set(CUDA_COMMON_DEBUG_FLAGS -restrict; -arch ${CUDA_ARCH}; -std c++11; --expt-extended-lambda; --expt-relaxed-constexpr)

set(HOST_OPT_FLAGS -Xcompiler -O3 -Xcompiler -qxlcompatmacros -Xcompiler -qlanglvl=extended0x  -Xcompiler -qalias=noansi -Xcompiler -qsmp=omp -Xcompiler -qhot -Xcompiler -qnoeh -Xcompiler -qsuppress=1500-029 -Xcompiler -qsuppress=1500-036)

set(HOST_RELDEB_FLAGS -Xcompiler -O2 -Xcompiler -g -Xcompiler -qstrict -Xcompiler -qsmp=omp:noopt -Xcompiler -qkeepparm -Xcompiler -qmaxmem=-1 -Xcompiler -qnoeh -Xcompiler -qsuppress=1500-029 -Xcompiler -qsuppress=1500-030 -Xcompiler -qsuppress=1500-036 )

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(COMB_NVCC_FLAGS -O3; ${CUDA_COMMON_OPT_FLAGS}; ${HOST_OPT_FLAGS} CACHE LIST "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(COMB_NVCC_FLAGS -g; -O2;  ${CUDA_COMMON_OPT_FLAGS};  ${HOST_RELDEB_FLAGS}  CACHE LIST "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(COMB_NVCC_FLAGS -g; -G; -O0; ${CUDA_COMMON_DEBUG_FLAGS}; ${HOST_RELDEB_FLAGS} CACHE LIST "")
endif()

