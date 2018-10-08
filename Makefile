
.PHONY: opt all clean debug setup_env

INC_DIR=include
OBJ_DIR=obj
LIB_DIR=lib
SRC_DIR=src

_DEPS=basic_mempool.hpp align.hpp mutex.hpp memory.hpp for_all.hpp profiling.hpp MeshData.hpp MeshInfo.hpp Box3d.hpp comm.hpp utils.hpp cuda_utils.hpp batch_launch.hpp persistent_launch.hpp MultiBuffer.hpp batch_utils.hpp CommFactory.hpp SetReset.hpp
DEPS=$(patsubst %,$(INC_DIR)/%,$(_DEPS))

_OBJ_OPT=comb_o.o batch_launch_o.o persistent_launch_o.o MultiBuffer_o.o
_OBJ_DEB=comb_g.o batch_launch_g.o persistent_launch_g.o MultiBuffer_g.o
_OBJ_DLINK_OPT=dlinked_o.o
_OBJ_DLINK_DEB=dlinked_g.o
OBJ_OPT=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_OPT))
OBJ_DEB=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_DEB))
OBJ_DLINK_OPT=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_DLINK_OPT))
OBJ_DLINK_DEB=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_DLINK_DEB))

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
CUDA_PATH=/usr/tce/packages/cuda/cuda-$(CUDA_VERSION)
CUDA_ARCH=sm_70

# clangcuda
CLANGCUDA_NAME=clangcuda
CLANGCUDA_VERSION=7.0.0
CLANGCUDA_EXEC=clang++
CLANGCUDA_PATH=/usr/workspace/wsrzd/burmark1/pozulp/spack/opt/spack/linux-rhel7-ppc64le/gcc-4.9.3/llvm-7.0.0-2udlapjrujevo6uqz3avu75f7bz3klpx
CLANGCUDA_EXTRA_FLAGS=
CLANGCUDA_OPT_OMP_FLAG=-fopenmp
CLANGCUDA_DEB_OMP_FLAG=-fopenmp

# # xlc
# CXX_NAME=xl
# CXX_VERSION=beta-2018.09.26
# CXX_EXEC=xlC
# CXX_MODULE_NAME=$(CXX_NAME)/$(CXX_VERSION)
# CXX_EXTRA_FLAGS=-qmaxmem=-1
# CXX_OPT_OMP_FLAG=-qsmp=omp
# CXX_DEB_OMP_FLAG=-qsmp=omp:noopt

# clang
CXX_NAME=clang
CXX_VERSION=coral-2018.08.08
CXX_EXEC=clang++
CXX_MODULE_NAME=$(CXX_NAME)/$(CXX_VERSION)
CXX_EXTRA_FLAGS=
CXX_OPT_OMP_FLAG=-fopenmp
CXX_DEB_OMP_FLAG=-fopenmp

# # gcc
# CXX_NAME=gcc
# CXX_VERSION=7.3.1
# CXX_EXEC=g++
# CXX_MODULE_NAME=$(CXX_NAME)/$(CXX_VERSION)
# CXX_EXTRA_FLAGS=
# CXX_OPT_OMP_FLAG=-fopenmp
# CXX_DEB_OMP_FLAG=-fopenmp


# set of options common to all builds
CXX_COMMON_DEFINES=

CXX_COMMON_FLAGS=-std=c++11 -m64 -I./$(INC_DIR)

CXX_COMMON_OPT_FLAGS=
CXX_COMMON_DEB_FLAGS=


# set base compiler options
CXX_MPI_COMPILER=/usr/tce/packages/$(MPI_NAME)/$(MPI_NAME)-$(MPI_VERSION)-$(CXX_NAME)-$(CXX_VERSION)/bin/$(MPI_EXEC)$(CXX_EXEC)

CXX_BASE_DEFINES=-DCOMB_COMPILER=$(CXX_MPI_COMPILER)

CXX_BASE_FLAGS=$(CXX_EXTRA_FLAGS)

CXX_BASE_OPT_FLAGS=-O3 -g
CXX_BASE_DEB_FLAGS=-O0 -g


# add openmp to compile options
# BEGIN COMMENT to disable openmp
CXX_COMMON_DEFINES+=-DCOMB_HAVE_OPENMP=1

CXX_BASE_OPT_FLAGS+=$(CXX_OPT_OMP_FLAG)
CXX_BASE_DEB_FLAGS+=$(CXX_DEB_OMP_FLAG)
# END COMMENT to disable openmp


# add cuda to compile options
CUDA_COMPILER=$(CUDA_PATH)/bin/$(CUDA_EXEC)
CXX_CUDA_COMPILER=$(CUDA_COMPILER)

CXX_CUDA_DEFINES=-DCOMB_HAVE_CUDA=1 -DCOMB_CUDA_COMPILER=$(CUDA_COMPILER)

CXX_CUDA_FLAGS=-rdc=true -arch=$(CUDA_ARCH) --expt-extended-lambda
CXX_CUDA_OPT_FLAGS=-O3 -g -lineinfo
CXX_CUDA_DEB_FLAGS=-O0 -g -G

LINK_CUDA_FLAGS=-lnvToolsExt


# clangcuda compile options
MPI_WRAPPER_CXX_FLAGS=$(shell $(CXX_MPI_COMPILER) -c $(SRC_DIR)/comb.cpp -o $(OBJ_DIR)/comb_o.o -showas | sed "s@.*$(OBJ_DIR)/comb_o\.o@@")
MPI_WRAPPER_LINK_FLAGS=$(shell $(CXX_MPI_COMPILER) $(OBJ_DIR)/comb_o.o -o comb -showas | sed "s@.*\ -o\ comb\ @@")

CLANGCUDA_COMPILER=$(CLANGCUDA_PATH)/bin/$(CLANGCUDA_EXEC)
CXX_CLANGCUDA_COMPILER=$(CLANGCUDA_COMPILER)

CXX_CLANGCUDA_DEFINES=-DCOMB_HAVE_CUDA=1 -DCOMB_CUDA_COMPILER=$(CLANGCUDA_COMPILER)

CXX_CLANGCUDA_FLAGS=$(CLANGCUDA_EXTRA_FLAGS) $(MPI_WRAPPER_CXX_FLAGS) -fcuda-rdc --cuda-gpu-arch=$(CUDA_ARCH)
CXX_CLANGCUDA_OPT_FLAGS=-O3 -g
CXX_CLANGCUDA_DEB_FLAGS=-O0 -g

LINK_CLANGCUDA_FLAGS=$(MPI_WRAPPER_LINK_FLAGS) -L$(CUDA_PATH)/lib64 -lcudart_static -lcudadevrt -lrt -ldl -pthread -lnvToolsExt -Wl,-rpath,$(CLANGCUDA_PATH)/lib



# combine flags into final form with cuda
# BEGIN SECTION to use cuda
CXX=$(CXX_CUDA_COMPILER) -x=cu -ccbin $(CXX_MPI_COMPILER)
CXX_FLAGS=$(CXX_COMMON_DEFINES) $(CXX_COMMON_FLAGS) $(CXX_BASE_DEFINES) $(CXX_CUDA_DEFINES) $(CXX_CUDA_FLAGS)
CXX_OPT_FLAGS=$(CXX_FLAGS) $(CXX_COMMON_OPT_FLAGS) -Xcompiler '$(CXX_BASE_FLAGS) $(CXX_BASE_OPT_FLAGS)' $(CXX_CUDA_OPT_FLAGS)
CXX_DEB_FLAGS=$(CXX_FLAGS) $(CXX_COMMON_DEB_FLAGS) -Xcompiler '$(CXX_BASE_FLAGS) $(CXX_BASE_DEB_FLAGS)' $(CXX_CUDA_DEB_FLAGS)
DLINK=$(CUDA_COMPILER) -dlink
DLINK_FLAGS=
DLINK_OPT_FLAGS=$(DLINK_FLAGS) $(CXX_OPT_FLAGS)
DLINK_DEB_FLAGS=$(DLINK_FLAGS) $(CXX_DEB_FLAGS)
LINK=$(CXX_CUDA_COMPILER) -ccbin $(CXX_MPI_COMPILER)
LINK_FLAGS=$(LINK_CUDA_FLAGS)
LINK_OPT_FLAGS=$(LINK_FLAGS) $(CXX_OPT_FLAGS)
LINK_DEB_FLAGS=$(LINK_FLAGS) $(CXX_DEB_FLAGS)
OBJ_LINK_OPT=$(OBJ_OPT) $(OBJ_DLINK_OPT)
OBJ_LINK_DEB=$(OBJ_DEB) $(OBJ_DLINK_DEB)
# END SECTION to use cuda

# combine flags into final form with clangcuda
# BEGIN SECTION to use cuda
# CXX=$(CXX_CLANGCUDA_COMPILER) -x cuda --cuda-path=$(CUDA_PATH) --gcc-toolchain=/usr/tce/packages/gcc/gcc-4.9.3
# CXX_FLAGS=$(CXX_COMMON_DEFINES) $(CXX_COMMON_FLAGS) $(CXX_CLANGCUDA_DEFINES) $(CXX_CLANGCUDA_FLAGS)
# CXX_OPT_FLAGS=$(CXX_FLAGS) $(CXX_COMMON_OPT_FLAGS) $(CXX_CLANGCUDA_OPT_FLAGS) $(CLANGCUDA_OPT_OMP_FLAG)
# CXX_DEB_FLAGS=$(CXX_FLAGS) $(CXX_COMMON_DEB_FLAGS) $(CXX_CLANGCUDA_DEB_FLAGS) $(CLANGCUDA_DEB_OMP_FLAG)
# DLINK=$(CUDA_COMPILER) -dlink
# DLINK_FLAGS=-arch=$(CUDA_ARCH)
# DLINK_OPT_FLAGS=$(DLINK_FLAGS)
# DLINK_DEB_FLAGS=$(DLINK_FLAGS)
# LINK=$(CXX_CLANGCUDA_COMPILER) --cuda-path=$(CUDA_PATH) --gcc-toolchain=/usr/tce/packages/gcc/gcc-4.9.3
# # link flags (-lcudart-static) must go after objects
# LINK_FLAGS=$(LINK_CLANGCUDA_FLAGS)
# LINK_OPT_FLAGS=$(LINK_FLAGS) $(CLANGCUDA_OPT_OMP_FLAG)
# LINK_DEB_FLAGS=$(LINK_FLAGS) $(CLANGCUDA_DEB_OMP_FLAG)
# OBJ_LINK_OPT=$(OBJ_OPT) $(OBJ_DLINK_OPT)
# OBJ_LINK_DEB=$(OBJ_DEB) $(OBJ_DLINK_DEB)
# END SECTION to use cuda

# combine flags into final form without cuda
# BEGIN SECTION to not use cuda
# CXX=$(CXX_MPI_COMPILER)
# CXX_FLAGS=$(CXX_COMMON_DEFINES) $(CXX_COMMON_FLAGS) $(CXX_BASE_DEFINES)
# CXX_OPT_FLAGS=$(CXX_FLAGS) $(CXX_COMMON_OPT_FLAGS) $(CXX_BASE_FLAGS) $(CXX_BASE_OPT_FLAGS)
# CXX_DEB_FLAGS=$(CXX_FLAGS) $(CXX_COMMON_DEB_FLAGS) $(CXX_BASE_FLAGS) $(CXX_BASE_DEB_FLAGS)
# DLINK=$(CXX_MPI_COMPILER)
# DLINK_FLAGS=
# DLINK_OPT_FLAGS=$(DLINK_FLAGS) $(CXX_OPT_FLAGS)
# DLINK_DEB_FLAGS=$(DLINK_FLAGS) $(CXX_DEB_FLAGS)
# LINK=$(CXX_MPI_COMPILER)
# LINK_FLAGS=
# LINK_OPT_FLAGS=$(LINK_FLAGS) $(CXX_OPT_FLAGS)
# LINK_DEB_FLAGS=$(LINK_FLAGS) $(CXX_DEB_FLAGS)
# OBJ_LINK_OPT=$(OBJ_OPT)
# OBJ_LINK_DEB=$(OBJ_DEB)
# END SECTION to not use cuda

opt: setup_env comb

all: setup_env comb comb_g

debug: setup_env comb_g

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

setup_env:
	echo "module load $(MPI_MODULE_NAME) $(CUDA_MODULE_NAME) $(CXX_MODULE_NAME)"


$(OBJ_DIR)/%_o.o: $(SRC_DIR)/%.cpp $(DEPS) $(OBJ_DIR)
	$(CXX) $(CXX_OPT_FLAGS) -c $< -o $@

$(OBJ_DLINK_OPT): $(OBJ_OPT)
	$(DLINK) $(DLINK_OPT_FLAGS) $^ -o $@

comb: $(OBJ_LINK_OPT)
	$(LINK) $^ -o $@ $(LINK_OPT_FLAGS)


$(OBJ_DIR)/%_g.o: $(SRC_DIR)/%.cpp $(DEPS) $(OBJ_DIR)
	$(CXX) $(CXX_DEB_FLAGS) -c $< -o $@

$(OBJ_DLINK_DEB): $(OBJ_DEB)
	$(DLINK) $(DLINK_DEB_FLAGS) $^ -o $@

comb_g: $(OBJ_LINK_DEB)
	$(LINK) $^ -o $@ $(LINK_DEB_FLAGS)


clean:
	rm -f $(OBJ_DIR)/*.o
