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

if (ENABLE_MPI)
  if(MPI_FOUND)
    message(STATUS "MPI Enabled")
  else()
    message(FATAL_ERROR "MPI NOT FOUND")
  endif()
endif()

if (ENABLE_OPENMP)
  if(OPENMP_FOUND)
    list(APPEND COMB_EXTRA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP Enabled")
  else()
    message(WARNING "OpenMP NOT FOUND")
    set(ENABLE_OPENMP Off)
  endif()
endif()


