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

if [[ $# -lt 2 ]]; then
  echo
  echo "You must pass 2 or more arguments to the script (in this order): "
  echo "   1) compiler version number"
  echo "   2) HIP compute architecture"
  echo "   3...) optional arguments to cmake"
  echo
  echo "For example: "
  echo "    toss4_amdclang.sh 4.1.0 gfx906"
  echo "    toss4_amdclang.sh 4.1.0 gfx906 -DBLT_CXX_STD=c++11"
  exit
fi

COMP_VER=$1
COMP_ARCH=$2
shift 2

MY_HIP_ARCH_FLAGS="--offload-arch=${COMP_ARCH}"
HOSTCONFIG="hip_X"

if [[ ${COMP_VER} == 4.5.* ]]
then
##HIP_CLANG_FLAGS="${MY_HIP_ARCH_FLAGS} -mllvm -amdgpu-fixed-function-abi=1"
  HOSTCONFIG="hip_4_link_X"
elif [[ ${COMP_VER} == 4.* ]]
then
  HOSTCONFIG="hip_4_link_X"
elif [[ ${COMP_VER} == 3.* ]]
then
  HOSTCONFIG="hip_X"
else
  echo "Unknown hip version, using ${HOSTCONFIG} host-config"
fi

BUILD_SUFFIX=lc_toss4-amdclang-${COMP_VER}-${COMP_ARCH}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

mkdir scripts && cd scripts && ln -s ../../scripts/*.bash . && ln -s ../bin/comb . && cd ..

module load cmake/3.14.5

# unload rocm to avoid configuration problems where the loaded rocm and COMP_VER
# are inconsistent causing the rocprim from the module to be used unexpectedly
module unload rocm


cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_CXX_COMPILER=/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.14-rocmcc-${COMP_VER}/bin/mpicxx \
  -DMPI_C_COMPILER=/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.14-rocmcc-${COMP_VER}/bin/mpicc \
  -DROCM_ROOT_DIR="/opt/rocm-${COMP_VER}" \
  -DHIP_ROOT_DIR="/opt/rocm-${COMP_VER}/hip" \
  -DHIP_CLANG_PATH=/opt/rocm-${COMP_VER}/llvm/bin \
  -DCMAKE_C_COMPILER=/opt/rocm-${COMP_VER}/llvm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-${COMP_VER}/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES="${MY_HIP_ARCH_FLAGS}" \
  -C "../host-configs/lc-builds/toss4/${HOSTCONFIG}.cmake" \
  -DENABLE_MPI=On \
  -DENABLE_HIP=On \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=Off \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build Comb"
echo
echo "  Please note that you have to have a consistent build environment"
echo "  when you make Comb as cmake may reconfigure; unload the rocm module"
echo "  or load the appropriate rocm module (${COMP_VER}) when building."
echo
echo "    module unload rocm"
echo "    srun -n1 make"
echo
echo "  Also note that libmodules.so is in the cce install. You may have to"
echo "  add that to your LD_LIBRARY_PATH to run."
echo
echo "    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/pe/cce/13.0.2/cce-clang/x86_64/lib:/opt/cray/pe/cce/13.0.2/cce/x86_64/lib"
echo "    srun -n1 ./bin/comb"
echo
echo "***********************************************************************"
