#!/usr/bin/env bash

##############################################################################
## Copyright (c) 2018-2021, Lawrence Livermore National Security, LLC.
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

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    toss3_icpc.sh 19.1.0"
  exit
fi

COMP_VER=$1
COMP_MAJOR_VER=${COMP_VER:0:2}
GCC_HEADER_VER=7
USE_TBB=On

if [ ${COMP_MAJOR_VER} -gt 18 ]
then
  GCC_HEADER_VER=8
fi

if [ ${COMP_MAJOR_VER} -lt 18 ]
then
  USE_TBB=Off
fi

BUILD_SUFFIX=lc_toss3-icpc-${COMP_VER}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_CXX_COMPILER=/usr/tce/packages/mvapich2/mvapich2-2.3-intel-${COMP_VER}/bin/mpic++ \
  -DMPI_C_COMPILER=/usr/tce/packages/mvapich2/mvapich2-2.3-intel-${COMP_VER}/bin/mpicc \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel/intel-${COMP_VER}/bin/icpc \
  -DCMAKE_C_COMPILER=/usr/tce/packages/intel/intel-${COMP_VER}/bin/icc \
  -DBLT_CXX_STD=c++11 \
  -C ../host-configs/lc-builds/toss3/icpc_X_gcc${GCC_HEADER_VER}headers.cmake \
  -DENABLE_MPI=On \
  -DENABLE_OPENMP=On \
  -DENABLE_TBB=${USE_TBB} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
