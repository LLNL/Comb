#!/bin/bash

##############################################################################
## Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
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

BUILD_SUFFIX=lc_blueos_mock_nvcc_dev_10_2_gcc_4_9_3

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.9.2 cuda-dev/10.2 gcc/4.9.3

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MPI=OFF \
  -DENABLE_OPENMP=ON \
  -DENABLE_CUDA=ON \
  -DCUDA_ARCH=sm_70 \
  -DCUDA_TOOLKIT_ROOT_DIR=/collab/usr/global/tools/nvidia/cuda/blueos_3_ppc64le_ib_p9/cuda-10.2.86 \
  -C ../host-configs/lc-builds/blueos/nvcc_gcc_4_9_3.cmake \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
