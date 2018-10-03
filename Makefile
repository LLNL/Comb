
.PHONY: opt all clean debug setup_env

INC_DIR=include
OBJ_DIR=obj
LIB_DIR=lib
SRC_DIR=src

# MPI
MPI_NAME=spectrum-mpi
MPI_VERSION=rolling-release
MPI_EXEC=mpi
MPI_MODULE_NAME=$(MPI_NAME)/$(MPI_VERSION)

# cuda
CUDA_NAME=cuda
CUDA_VERSION=9.2.148
CUDA_EXEC=nvcc
CUDA_MODULE_NAME=cuda/$(CUDA_VERSION)

# # xlc
# CXX_NAME=xl
# CXX_VERSION=beta-2018.09.26
# CXX_EXEC=xlC
# CXX_MODULE_NAME=$(CXX_NAME)/$(CXX_VERSION)
# CXX_EXTRA_FLAGS=-Xcompiler '-qmaxmem=-1'
# CXX_OPT_OMP_FLAG=-Xcompiler '-qsmp=omp'
# CXX_DEB_OMP_FLAG=-Xcompiler '-qsmp=omp:noopt'

# clang
CXX_NAME=clang
CXX_VERSION=coral-2018.08.08
CXX_EXEC=clang++
CXX_MODULE_NAME=$(CXX_NAME)/$(CXX_VERSION)
CXX_EXTRA_FLAGS=
#CXX_OPT_OMP_FLAG=-Xcompiler '-fopenmp'
#CXX_DEB_OMP_FLAG=-Xcompiler '-fopenmp'
CXX_OPT_OMP_FLAG=-fopenmp
CXX_DEB_OMP_FLAG=-fopenmp

# # gcc
# CXX_NAME=gcc
# CXX_VERSION=7.3.1
# CXX_EXEC=g++
# CXX_MODULE_NAME=$(CXX_NAME)/$(CXX_VERSION)
# CXX_EXTRA_FLAGS=
# CXX_OPT_OMP_FLAG=-Xcompiler '-fopenmp'
# CXX_DEB_OMP_FLAG=-Xcompiler '-fopenmp'


CUDA_COMPILER=/usr/tce/packages/cuda/cuda-$(CUDA_VERSION)/bin/$(CUDA_EXEC)
CXX_MPI_COMPILER=/usr/tce/packages/$(MPI_NAME)/$(MPI_NAME)-$(MPI_VERSION)-$(CXX_NAME)-$(CXX_VERSION)/bin/$(MPI_EXEC)$(CXX_EXEC)


CXX_DEFINES=-DCOMB_CUDA_COMPILER=$(CUDA_COMPILER) -DCOMB_COMPILER=$(CXX_MPI_COMPILER)

# CXX=$(CUDA_COMPILER) -ccbin $(CXX_MPI_COMPILER)
CXX=/usr/workspace/wsrzd/burmark1/pozulp/spack/opt/spack/linux-rhel7-ppc64le/gcc-4.9.3/llvm-7.0.0-2udlapjrujevo6uqz3avu75f7bz3klpx/bin/clang++
# CXX=/usr/workspace/wsrzc/pozulp1/sp/spack/opt/spack/linux-rhel7-ppc64le/gcc-4.9.3/llvm-7.0.0-2udlapjrujevo6uqz3avu75f7bz3klpx/bin/clang++
MPI_WRAPPER_FLAGS=-I/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-rolling-release/include -pthread -L/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-rolling-release/lib -lmpiprofilesupport -lmpi_ibm -Wl,-rpath,/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-rolling-release/lib

#CXX_FLAGS=$(CXX_EXTRA_FLAGS) -std=c++11 -I./$(INC_DIR) -lnvToolsExt -rdc=true -arch=sm_60 --expt-extended-lambda -m64 $(CXX_DEFINES)
CXX_FLAGS=$(MPI_WRAPPER_FLAGS) -std=c++11 -I./$(INC_DIR) -L/usr/tce/packages/cuda/cuda-9.2.148/lib64 -lnvToolsExt -fcuda-rdc --cuda-gpu-arch=sm_70 -m64 $(CXX_DEFINES) -Wl,-rpath,/usr/workspace/wsrzd/burmark1/pozulp/spack/opt/spack/linux-rhel7-ppc64le/gcc-4.9.3/llvm-7.0.0-2udlapjrujevo6uqz3avu75f7bz3klpx/lib
#CXX_OPT_FLAGS=$(CXX_FLAGS) $(CXX_OPT_OMP_FLAG) -O3 -g -lineinfo  -Xcompiler '-O3 -g'
#CXX_DEB_FLAGS=$(CXX_FLAGS) $(CXX_DEB_OMP_FLAG) -O0 -G -g -Xcompiler '-O0 -g'
CXX_OPT_FLAGS=$(CXX_FLAGS) $(CXX_OPT_OMP_FLAG) -O3 -g
CXX_DEB_FLAGS=$(CXX_FLAGS) $(CXX_DEB_OMP_FLAG) -O0 -G -g

_DEPS=basic_mempool.hpp align.hpp mutex.hpp memory.cuh for_all.cuh profiling.cuh MeshData.cuh MeshInfo.cuh Box3d.cuh comm.cuh utils.cuh cuda_utils.cuh batch_launch.cuh persistent_launch.cuh MultiBuffer.cuh batch_utils.cuh CommFactory.cuh SetReset.cuh batch_exec.cuh
DEPS=$(patsubst %,$(INC_DIR)/%,$(_DEPS))

_OBJ_OPT=comb_o.o batch_launch_o.o persistent_launch_o.o MultiBuffer_o.o
_OBJ_DEB=comb_g.o batch_launch_g.o persistent_launch_g.o MultiBuffer_g.o
OBJ_OPT=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_OPT))
OBJ_DEB=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_DEB))

opt: setup_env comb_o

all: setup_env comb_o comb_g

debug: setup_env comb_g

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

#setup_env:
#	echo "module load $(MPI_MODULE_NAME) $(CUDA_MODULE_NAME) $(CXX_MODULE_NAME)"


$(OBJ_DIR)/%_o.o: $(SRC_DIR)/%.cu $(DEPS) $(OBJ_DIR)
	$(CXX) $(CXX_OPT_FLAGS) -c $< -o $@

$(OBJ_DIR)/dlinked_o.o: $(OBJ_OPT)
	nvcc -arch=sm_70 -dlink $^ -o $(OBJ_DIR)/dlinked_o.o

comb_o: $(OBJ_OPT) $(OBJ_DIR)/dlinked_o.o
	$(CXX) $(CXX_OPT_FLAGS) $^ -o $@ -lcudart_static -lcudadevrt -lrt -ldl


$(OBJ_DIR)/%_g.o: $(SRC_DIR)/%.cu $(DEPS) $(OBJ_DIR)
	$(CXX) $(CXX_DEB_FLAGS) -c $< -o $@

$(OBJ_DIR)/dlinked_g.o: $(OBJ_DEB)
	nvcc -arch=sm_70 -dlink $^ -o $(OBJ_DIR)/dlinked_g.o

comb_g: $(OBJ_DEB) $(OBJ_DIR)/dlinked_g.o
	$(CXX) $(CXX_DEB_FLAGS) $^ -o $@ -lcudart_static -lcudadevrt -lrt -ldl


clean:
	rm -f $(OBJ_DIR)/*.o
