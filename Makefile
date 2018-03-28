
.PHONY: opt all clean debug

CXX=nvcc -ccbin mpixlC -Xcompiler '-qmaxmem=-1'
#CXX=nvcc -ccbin mpiclang++
#CXX=nvcc -ccbin mpig++
CXX_FLAGS=-std=c++14 -I. -lnvToolsExt -rdc=true -arch=sm_70 --expt-extended-lambda
CXX_OPT_FLAGS=$(CXX_FLAGS) -O2 -g -lineinfo  -Xcompiler '-O2 -g'
CXX_DEBUG_FLAGS=$(CXX_FLAGS) -O0 -g -G -Xcompiler '-O0 -g'

DEPS=basic_mempool.hpp align.hpp mutex.hpp memory.cuh for_all.cuh profiling.cuh mesh.cuh comm.cuh utils.cuh batch_launch.cuh persistent_launch.cuh
OBJ_OPT=test_comm_o.o batch_launch_o.o persistent_launch_o.o
OBJ_DEBUG=test_comm_g.o batch_launch_g.o persistent_launch_g.o

opt: test_comm_o

all: test_comm_o test_comm_g

debug: test_comm_g

%_o.o: %.cu $(DEPS)
	$(CXX) $(CXX_OPT_FLAGS) -c $< -o $@

%_g.o: %.cu $(DEPS)
	$(CXX) $(CXX_DEBUG_FLAGS) -c $< -o $@

test_comm_o: $(OBJ_OPT)
	$(CXX) $(CXX_OPT_FLAGS) $^ -o $@

test_comm_g: $(OBJ_DEBUG)
	$(CXX) $(CXX_DEBUG_FLAGS) $^ -o $@

clean:
	rm -f *.o

