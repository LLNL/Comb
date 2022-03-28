#!/usr/bin/env bash

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

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    blueos_pgi.sh 21.1"
  exit
fi

COMP_VER=$1
shift 1

BUILD_SUFFIX=lc_blueos-pgi-${COMP_VER}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_CXX_COMPILER=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-pgi-${COMP_VER}/bin/mpipgc++ \
  -DMPI_C_COMPILER=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-pgi-${COMP_VER}/bin/mpipgcc \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/pgi/pgi-${COMP_VER}/bin/pgc++ \
  -DCMAKE_C_COMPILER=/usr/tce/packages/pgi/pgi-${COMP_VER}/bin/pgcc \
  -C ../host-configs/lc-builds/blueos/pgi_X.cmake \
  -DENABLE_MPI=On \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
