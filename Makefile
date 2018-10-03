
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


# set basic compiler options
CXX_MPI_COMPILER=/usr/tce/packages/$(MPI_NAME)/$(MPI_NAME)-$(MPI_VERSION)-$(CXX_NAME)-$(CXX_VERSION)/bin/$(MPI_EXEC)$(CXX_EXEC)

CXX_COMMON_DEFINES=-DCOMB_COMPILER=$(CXX_MPI_COMPILER)

CXX_COMMON_FLAGS=-std=c++11 -m64 -I./$(INC_DIR)

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
CUDA_COMPILER=/usr/tce/packages/cuda/cuda-$(CUDA_VERSION)/bin/$(CUDA_EXEC)
CXX_CUDA_COMPILER=$(CUDA_COMPILER) -ccbin $(CXX_MPI_COMPILER)

CXX_CUDA_DEFINES=-DCOMB_HAVE_CUDA=1 -DCOMB_CUDA_COMPILER=$(CUDA_COMPILER)

CXX_CUDA_FLAGS=-lnvToolsExt -rdc=true -arch=sm_60 --expt-extended-lambda
CXX_CUDA_OPT_FLAGS=-O3 -g -lineinfo
CXX_CUDA_DEB_FLAGS=-O0 -g -G


# combine flags into final form with cuda
# BEGIN COMMENT to disable cuda
CXX_FLAGS=$(CXX_COMMON_FLAGS) $(CXX_COMMON_DEFINES) $(CXX_CUDA_DEFINES) $(CXX_CUDA_FLAGS)
CXX_OPT_FLAGS=$(CXX_FLAGS) -Xcompiler '$(CXX_BASE_FLAGS) $(CXX_BASE_OPT_FLAGS)' $(CXX_CUDA_OPT_FLAGS)
CXX_DEB_FLAGS=$(CXX_FLAGS) -Xcompiler '$(CXX_BASE_FLAGS) $(CXX_BASE_DEB_FLAGS)' $(CXX_CUDA_DEB_FLAGS)
CXX=$(CXX_CUDA_COMPILER) -x=cu
DLINK=$(CXX_CUDA_COMPILER) -dlink
LINK=$(CXX_CUDA_COMPILER)
OBJ_LINK_OPT=$(OBJ_OPT) $(OBJ_DLINK_OPT)
OBJ_LINK_DEB=$(OBJ_DEB) $(OBJ_DLINK_DEB)
# END COMMENT to disable cuda

# combine flags into final form without cuda
# BEGIN COMMENT to enable cuda
# CXX_FLAGS=$(CXX_COMMON_FLAGS) $(CXX_COMMON_DEFINES)
# CXX_OPT_FLAGS=$(CXX_FLAGS) $(CXX_BASE_FLAGS) $(CXX_BASE_OPT_FLAGS)
# CXX_DEB_FLAGS=$(CXX_FLAGS) $(CXX_BASE_FLAGS) $(CXX_BASE_DEB_FLAGS)
# CXX=$(CXX_MPI_COMPILER)
# DLINK=$(CXX_MPI_COMPILER)
# LINK=$(CXX_MPI_COMPILER)
# OBJ_LINK_OPT=$(OBJ_OPT)
# OBJ_LINK_DEB=$(OBJ_DEB)
# END COMMENT to enable cuda

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
	$(DLINK) $(CXX_OPT_FLAGS) $^ -o $@

comb: $(OBJ_LINK_OPT)
	$(LINK) $(CXX_OPT_FLAGS) $^ -o $@


$(OBJ_DIR)/%_g.o: $(SRC_DIR)/%.cpp $(DEPS) $(OBJ_DIR)
	$(CXX) $(CXX_DEB_FLAGS) -c $< -o $@

$(OBJ_DLINK_DEB): $(OBJ_DEB)
	$(DLINK) $(CXX_DEB_FLAGS) $^ -o $@

comb_g: $(OBJ_LINK_DEB)
	$(LINK) $(CXX_DEB_FLAGS) $^ -o $@


clean:
	rm -f $(OBJ_DIR)/*.o
