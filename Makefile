
.PHONY: opt all clean debug

INC_DIR=include
OBJ_DIR=obj
LIB_DIR=lib
SRC_DIR=src

# CXX=nvcc -ccbin mpixlC -Xcompiler '-qmaxmem=-1'
CXX=nvcc -ccbin mpiclang++
#CXX=nvcc -ccbin mpig++
CXX_FLAGS=-std=c++11 -I./$(INC_DIR) -lnvToolsExt -rdc=true -arch=sm_60 --expt-extended-lambda -m64
CXX_OPT_FLAGS=$(CXX_FLAGS) -O2 -g -lineinfo  -Xcompiler '-O2 -g'
CXX_DEB_FLAGS=$(CXX_FLAGS) -O0 -G -g -Xcompiler '-O0 -g'

_DEPS=basic_mempool.hpp align.hpp mutex.hpp memory.cuh for_all.cuh profiling.cuh MeshData.cuh MeshInfo.cuh Box3d.cuh comm.cuh utils.cuh cuda_utils.cuh batch_launch.cuh persistent_launch.cuh MultiBuffer.cuh batch_utils.cuh CommFactory.cuh
DEPS=$(patsubst %,$(INC_DIR)/%,$(_DEPS))

_OBJ_OPT=comb_o.o batch_launch_o.o persistent_launch_o.o MultiBuffer_o.o
_OBJ_DEB=comb_g.o batch_launch_g.o persistent_launch_g.o MultiBuffer_g.o
OBJ_OPT=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_OPT))
OBJ_DEB=$(patsubst %,$(OBJ_DIR)/%,$(_OBJ_DEB))

opt: comb_o

all: comb_o comb_g

debug: comb_g


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
