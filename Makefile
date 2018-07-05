
.PHONY: opt all clean debug setup_env

INC_DIR=include
OBJ_DIR=obj
LIB_DIR=lib
SRC_DIR=src

# MPI
MPI_NAME=spectrum-mpi
MPI_VERSION=2018.06.07
MPI_MODULE_NAME=$(MPI_NAME)/$(MPI_VERSION)

# cuda
CUDA_COMPILER_NAME=nvcc
CUDA_COMPILER_VERSION=9.2.88
CUDA_MODULE_NAME=cuda/$(CUDA_COMPILER_VERSION)

# # xlc
# CXX_COMPILER_NAME=xl
# CXX_COMPILER_VERSION=beta-2018.06.27
# CXX_COMPILER_EXEC_NAME=xlC
# CXX_EXTRA_FLAGS=-Xcompiler '-qmaxmem=-1'
# CXX_OPT_OMP_FLAG=-Xcompiler '-qsmp=omp'
# CXX_DEB_OMP_FLAG=-Xcompiler '-qsmp=omp:noopt'

# clang
CXX_COMPILER_NAME=clang
CXX_COMPILER_VERSION=coral-2018.05.23
CXX_COMPILER_EXEC_NAME=clang++
CXX_EXTRA_FLAGS=
CXX_OPT_OMP_FLAG=-Xcompiler '-fopenmp'
CXX_DEB_OMP_FLAG=-Xcompiler '-fopenmp'

# # gcc
# CXX_COMPILER_NAME=gcc
# CXX_COMPILER_VERSION=7.2.1-redhat
# CXX_COMPILER_EXEC_NAME=g++
# CXX_EXTRA_FLAGS=
# CXX_OPT_OMP_FLAG=-Xcompiler '-fopenmp'
# CXX_DEB_OMP_FLAG=-Xcompiler '-fopenmp'


CUDA_COMPILER=/usr/tce/packages/cuda/cuda-$(CUDA_COMPILER_VERSION)/bin/nvcc
CXX_MPI_COMPILER=/usr/tce/packages/$(MPI_NAME)/$(MPI_NAME)-$(MPI_VERSION)-$(CXX_COMPILER_NAME)-$(CXX_COMPILER_VERSION)/bin/mpi$(CXX_COMPILER_EXEC_NAME)


CXX_DEFINES=-DCOMB_CUDA_COMPILER=$(CUDA_COMPILER) -DCOMB_COMPILER=$(CXX_MPI_COMPILER)

CXX=$(CUDA_COMPILER) -ccbin $(CXX_MPI_COMPILER)

CXX_FLAGS=$(CXX_EXTRA_FLAGS) -std=c++11 -I./$(INC_DIR) -lnvToolsExt -rdc=true -arch=sm_60 --expt-extended-lambda -m64 $(CXX_DEFINES)
CXX_OPT_FLAGS=$(CXX_FLAGS) $(CXX_OPT_OMP_FLAG) -O3 -g -lineinfo  -Xcompiler '-O3 -g'
CXX_DEB_FLAGS=$(CXX_FLAGS) $(CXX_DEB_OMP_FLAG) -O0 -G -g -Xcompiler '-O0 -g'

_DEPS=basic_mempool.hpp align.hpp mutex.hpp memory.cuh for_all.cuh profiling.cuh MeshData.cuh MeshInfo.cuh Box3d.cuh comm.cuh utils.cuh cuda_utils.cuh batch_launch.cuh persistent_launch.cuh MultiBuffer.cuh batch_utils.cuh CommFactory.cuh SetReset.cuh
DEPS=$(patsubst %,$(INC_DIR)/%,$(_DEPS))

_OBJ_OPT=comb_o.o batch_launch_o.o persistent_launch_o.o MultiBuffer_o.o
_OBJ_DEB=comb_g.o batch_launch_g.o persistent_launch_g.o MultiBuffer_g.o
OBJ_OPT=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_OPT))
OBJ_DEB=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_DEB))

opt: comb_o setup_env

all: comb_o comb_g setup_env

debug: comb_g setup_env


setup_env:
	echo "module load $(MPI_MODULE_NAME) $(CUDA_MODULE_NAME)"


$(OBJ_DIR)/%_o.o: $(SRC_DIR)/%.cu $(DEPS)
	$(CXX) $(CXX_OPT_FLAGS) -c $< -o $@

comb_o: $(OBJ_OPT)
	$(CXX) $(CXX_OPT_FLAGS) $^ -o $@


$(OBJ_DIR)/%_g.o: $(SRC_DIR)/%.cu $(DEPS)
	$(CXX) $(CXX_DEB_FLAGS) -c $< -o $@

comb_g: $(OBJ_DEB)
	$(CXX) $(CXX_DEB_FLAGS) $^ -o $@


clean:
	rm -f $(OBJ_DIR)/*.o
