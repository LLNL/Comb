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

##
## Here are the CMake dependent options in COMB.
##

cmake_dependent_option(COMB_ENABLE_MPI "Build MPI support" On "ENABLE_MPI" Off)
cmake_dependent_option(COMB_ENABLE_OPENMP "Build OpenMP support" On "ENABLE_OPENMP" Off)
cmake_dependent_option(COMB_ENABLE_CUDA "Build CUDA support" On "ENABLE_CUDA" Off)
cmake_dependent_option(COMB_ENABLE_HIP "Build HIP support" On "ENABLE_HIP" Off)
cmake_dependent_option(COMB_ENABLE_CLANG_CUDA "Build Clang CUDA support" On "ENABLE_CLANG_CUDA" Off)

cmake_dependent_option(COMB_ENABLE_NV_TOOLS_EXT "Build NV_TOOLS_EXT support" On "COMB_ENABLE_CUDA" Off)
cmake_dependent_option(COMB_ENABLE_ROCTX "Build ENABLE_ROCTX support" On "COMB_ENABLE_HIP" Off)
