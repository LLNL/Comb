#!/bin/bash

### TODO: make sure the COMB_PATH is set correctly;
COMB_PATH=$PWD

### build COMB with MVAPIH2 (or other MPI libraries)
module unload spectrum-mpi
module load cmake/3.9.2 cuda/9.2.148

MPI_NAME=MVAPICH2-GDR-jsrun
USE_MVAPICH2_RSH=0

if [[ ! "x" == "x$SYS_TYPE" ]]; then
    if [[ "x$SYS_TYPE" =~ xblueos.*_p9 ]]; then
        export MODULEPATH="/usr/tcetmp/modulefiles/Compiler:$MODULEPATH"
        module load gcc/7.3.1
        module load gcc/7.3.1/mvapich2-gdr/2.3.2-cuda9.2.148-jsrun
        if [[ $USE_MVAPICH2_RSH -eq 1 ]]; then
            MPI_NAME=MVAPICH2-GDR
            # use mpirun_rsh as the launcher instead of jsrun
            module load gcc/7.3.1/mvapich2-gdr/2.3.2-cuda9.2.148
        fi
    else
        module load gcc/7.4.0
    fi
fi

### TODO: add installation path of other version of MVAPICH2 library (or other MPI libraries) if desired
MPI_HOME=
if [[ ! $MPI_HOME == "" ]]; then
    export PATH=$MPI_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
fi

BUILD_SUFFIX=nvcc_9_2_gcc_7_3_1_${MPI_NAME}
rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

echo "Building COMB with ${MPI_NAME}..."

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENMP=OFF \
  -DENABLE_CUDA=ON \
  -DCUDA_ARCH=sm_70 \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

make
cd -

