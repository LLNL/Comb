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

# Enable MPI by by default
set(ENABLE_MPI On CACHE BOOL "Build MPI support")

# Build options
set(COMB_ENABLE_GDSYNC Off CACHE BOOL "Build GDSYNC support")
set(COMB_ENABLE_GPUMP Off CACHE BOOL "Build GPUMP support")
set(COMB_ENABLE_MP Off CACHE BOOL "Build MP support")
set(COMB_ENABLE_UMR Off CACHE BOOL "Build UMR support")
set(COMB_ENABLE_RAJA ON CACHE BOOL "Build RAJA support")
set(COMB_ENABLE_CALIPER Off CACHE BOOL "Build Caliper support")
set(COMB_ENABLE_ADIAK Off CACHE BOOL "Build Adiak support")

option(COMB_ENABLE_LOG "Build logging support" Off)

# Build options for libraries, disable extras
option(ENABLE_TESTS "Build tests" Off)
option(ENABLE_REPRODUCERS "Build issue reproducers" Off)
option(ENABLE_EXAMPLES "Build simple examples" Off)
option(ENABLE_EXERCISES "Build exercises " Off)
option(ENABLE_MODULES "Enable modules in supporting compilers (clang)" Off)

if (ENABLE_CUDA)
  # Separable compilation is required by comb, set before load BLT
  set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "")
  if (NOT DEFINED CUDA_ARCH)
    message(STATUS "CUDA compute architecture set to Comb default sm_35 since it was not specified")
    set(CUDA_ARCH "sm_35" CACHE STRING "Set CUDA_ARCH to Comb minimum supported" FORCE)
  endif()
endif()

if (ENABLE_HIP)
  # Separable compilation is required by comb, set before load BLT
  # set(HIP_SEPARABLE_COMPILATION ON CACHE BOOL "")
endif()
