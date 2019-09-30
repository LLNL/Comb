#!/bin/bash

module load cuda/9.2.148

COMB_PATH=$PWD
### TODO: output files will be generated on RUN_PATH
RUN_PATH=$PWD
cd $COMB_PATH

procs_per_side=2

run_smpi=0
run_mv2=1

# Test COMB with SpectrumMPI
if [ $run_smpi -eq 1 ]; then
    MPI_NAME=SpectrumMPI
    BUILD_PATH=$COMB_PATH/build_lc_blueos_nvcc_9_2_gcc_4_9_3
    # build the comb if executable does not exist
    # TODO: have a generic script (now only support on Lassen)
    if [ ! -f "$BUILD_PATH/bin/comb" ]; then
        echo "Building COMB with SpectrumMPI..."
        $COMB_PATH/scripts/lc-builds/blueos/nvcc_9_2_gcc_4_9_3.sh
        cd $COMB_PATH/build_lc_blueos_nvcc_9_2_gcc_4_9_3
        make
    fi
    # run the test
    if [ ! -f "$BUILD_PATH/bin/comb" ]; then
        echo "failed to build COMB with $MPI_NAME"
    else
        TEST_PATH=$RUN_PATH/${procs_per_side}_${procs_per_side}_${procs_per_side}_$MPI_NAME
        rm -fr $TEST_PATH
        mkdir -p $TEST_PATH
        cd $TEST_PATH
        ln -s $BUILD_PATH/bin/comb .
        ln -s $COMB_PATH/scripts/* .

        echo "Running COMB with ${MPI_NAME}..."
        #./run_tests.bash 2 $TEST_PATH/focused_tests.bash
        ./run_tests.bash ${procs_per_side} $TEST_PATH/focused_mpi_type_tests.bash
    fi
fi

### Test COMB with MVAPICH2-GDR
if [ $run_mv2 -eq 1 ]; then
    cd $COMB_PATH
    module unload spectrum-mpi
    export MODULEPATH="/usr/tcetmp/modulefiles/Compiler:$MODULEPATH"
    module load gcc/7.3.1 gcc/7.3.1/mvapich2-gdr/2.3.2-cuda9.2.148-jsrun

    ### TODO: set USE_MVAPICH2_RSH to 1 to use mpirun_rsh instead of jsrun/lrun
    ###       for systems where jsrun is not available, e.g., x86
    export USE_MVAPICH2_RSH=0
    if [[ $USE_MVAPICH2_RSH -eq 1 ]]; then
        MPI_NAME=MVAPICH2-GDR
        module load gcc/7.3.1/mvapich2-gdr/2.3.2-cuda9.2.148
        ### TODO: RSH version of MVAPICH2 should have different path to jsrun version
        export MPI_HOME=/usr/tcetmp/packages/mvapich2/mvapich2-gdr-2.3.2-gcc-7.3.1-cuda9.2.148-2019-08-09
    else
        MPI_NAME=MVAPICH2-GDR-jsrun
        ### TODO: Add PMI4PMIX library if needed
        PMI4_PATH=/usr/global/tools/pmi4pmix/blueos_3_ppc64le_ib
        export LD_LIBRARY_PATH=$PMI4_PATH/lib:$LD_LIBRARY_PATH
        ### TODO: Add path of MVAPICH2 library built with jsrun launcher (or other MPI libraries)
        MPI_HOME=/usr/tcetmp/packages/mvapich2/mvapich2-gdr-2.3.2-gcc-7.3.1-cuda9.2.148-2019-08-09-jsrun
        export OMPI_LD_PRELOAD_PREPEND=$MPI_HOME/lib64/libmpi.so
    fi

    ### TODO: add installation path of other version of MVAPICH2 library (or other MPI libraries) if desired
    # MPI_NAME=MY-MVAPICH2-GDR-jsrun
    MPI_HOME=
    if [[ ! $MPI_HOME == "" ]]; then
        export MPI_HOME=$MPI_HOME
        export PATH=$MPI_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
        export OMPI_LD_PRELOAD_PREPEND=$MPI_HOME/lib/libmpi.so
    fi

    ### check if COMB executable exists
    BUILD_PATH=$COMB_PATH/build_nvcc_9_2_gcc_7_3_1_$MPI_NAME
    if [ ! -f "$BUILD_PATH/bin/comb" ]; then
        echo "Error: COMB build with $MPI_NAME is not found in $BUILD_PATH"
        exit;
    fi

    TEST_PATH=$RUN_PATH/${procs_per_side}_${procs_per_side}_${procs_per_side}_$MPI_NAME
    rm -fr $TEST_PATH
    mkdir -p $TEST_PATH
    cd $TEST_PATH
    ln -s $BUILD_PATH/bin/comb .
    ln -s $COMB_PATH/scripts/* .

    mv2_envs=$mv2_envs" MV2_USE_CUDA=1 MV2_USE_GDR=1 MV2_USE_GPUDIRECT_GDRCOPY=0 MV2_SHOW_CPU_BINDING=1 MV2_CPU_BINDING_POLICY=hybrid MV2_HYBRID_BINDING_POLICY=spread MV2_IBA_HCA=mlx5_0:mlx5_3"
    export USE_MVAPICH2=1
    export $mv2_envs
    ### TODO: change visible devices if desired
    #export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

    module list
    echo "Running COMB with ${MPI_NAME}..."
    ./run_tests.bash ${procs_per_side} $TEST_PATH/focused_mpi_type_tests.bash
fi
