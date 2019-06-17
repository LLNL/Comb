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

# Set up COMB_ENABLE prefixed options
set(COMB_ENABLE_OPENMP ${ENABLE_OPENMP})
set(COMB_ENABLE_CUDA ${ENABLE_CUDA})
set(COMB_ENABLE_CLANG_CUDA ${ENABLE_CLANG_CUDA})

# Configure a header file with all the variables we found.
configure_file(${PROJECT_SOURCE_DIR}/include/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/config.hpp)
