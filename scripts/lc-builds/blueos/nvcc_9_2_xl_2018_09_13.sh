#!/bin/bash

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

BUILD_SUFFIX=lc_blueos_nvcc_9_2_xl_2018_09_13

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.9.2 xl/2018.09.13 cuda/9.2.148

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/lc-builds/blueos/nvcc_xl_2018_09_13.cmake \
  -DENABLE_OPENMP=ON \
  -DENABLE_CUDA=ON \
  -DCUDA_ARCH=sm_60 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-9.2.148 \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
