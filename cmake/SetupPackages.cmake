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

if (ENABLE_MPI)
  if(MPI_FOUND)
    message(STATUS "MPI Enabled")
  else()
    message(FATAL_ERROR "MPI NOT FOUND")
  endif()
endif()


if (ENABLE_OPENMP)
  if(OPENMP_FOUND)
    message(STATUS "OpenMP Enabled")
  else()
    message(FATAL_ERROR "OpenMP NOT FOUND")
  endif()
endif()


if (ENABLE_CUDA)
  if(CUDA_FOUND)
    message(STATUS "Cuda Enabled")
  else()
    message(FATAL_ERROR "Cuda NOT FOUND")
  endif()
endif()
