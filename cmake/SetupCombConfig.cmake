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

# Set up COMB_ENABLE prefixed options
set(COMB_ENABLE_MPI ${ENABLE_MPI})
set(COMB_ENABLE_OPENMP ${ENABLE_OPENMP})
set(COMB_ENABLE_CUDA ${ENABLE_CUDA})
set(COMB_ENABLE_NV_TOOLS_EXT ${ENABLE_NV_TOOLS_EXT})
set(COMB_ENABLE_CLANG_CUDA ${ENABLE_CLANG_CUDA})
set(COMB_ENABLE_HIP ${ENABLE_HIP})
set(COMB_ENABLE_ROCTX ${ENABLE_ROCTX})
set(COMB_ENABLE_GDSYNC ${ENABLE_GDSYNC})
set(COMB_ENABLE_GPUMP ${ENABLE_GPUMP})
set(COMB_ENABLE_MP ${ENABLE_MP})
set(COMB_ENABLE_UMR ${ENABLE_UMR})
set(COMB_ENABLE_RAJA ${ENABLE_RAJA})
set(COMB_ENABLE_CALIPER ${ENABLE_CALIPER})
set(COMB_ENABLE_ADIAK ${ENABLE_ADIAK})

if (COMB_ENABLE_CUDA)
  if(CUDA_VERSION VERSION_GREATER_EQUAL 10)
    set(COMB_ENABLE_CUDA_GRAPH On)
  else()
    set(COMB_ENABLE_CUDA_GRAPH Off)
  endif()
endif()

set(COMB_CXX_COMPILER ${CMAKE_CXX_COMPILER})
set(COMB_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})
set(COMB_HIP_COMPILER ${CMAKE_HIP_CLANG_COMPILER})

# Configure a header file with all the variables we found.
configure_file(${PROJECT_SOURCE_DIR}/include/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/config.hpp)
