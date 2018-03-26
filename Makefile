
.PHONY: opt all clean debug

CXX=nvcc -ccbin mpixlC -Xcompiler -qmaxmem=-1 -I. -lnvToolsExt
# CXX=nvcc -ccbin mpiclang++ -I. -lnvToolsExt
CXX_OPT_FLAGS=-std=c++14 -O2 -g -lineinfo -arch=sm_70 --expt-extended-lambda -Xcompiler '-O2 -g'
CXX_DEBUG_FLAGS=-std=c++14 -O0 -g -G -arch=sm_70 --expt-extended-lambda -Xcompiler '-O0 -g'

DEPS=basic_mempool.hpp align.hpp mutex.hpp
OBJ_OPT=test_comm_o.o
OBJ_DEBUG=test_comm_g.o

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

