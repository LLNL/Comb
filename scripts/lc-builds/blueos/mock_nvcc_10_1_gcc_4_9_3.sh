#!/bin/bash

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

BUILD_SUFFIX=lc_blueos_mock_nvcc_10_1_gcc_4_9_3

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.9.2 cuda/10.1.243

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MPI=OFF \
  -DENABLE_OPENMP=ON \
  -DENABLE_CUDA=ON \
  -DCUDA_ARCH=sm_70 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-10.1.243 \
  -C ../host-configs/lc-builds/blueos/nvcc_gcc_4_9_3.cmake \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
