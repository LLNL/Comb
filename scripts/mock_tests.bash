#!/bin/bash

# This script should only be used with comb built without mpi

# Note: you may need to bind processes to cores to get reasonable openmp behavior
# Your scheduler may help with this
# Otherwise you may need to set environment variables for each proc to bind it to cores/threads
# http://www.nersc.gov/users/software/programming-models/openmp/process-and-thread-affinity/
# Ex:
#   bash:
#     mpirun -np 1 bind_script comb
#   bind_script:
#     export OMP_PLACES={0,2} # this depends on the local rank of the process if running more than one process per node
#     exec $@

# Comb executable or symlink
run_comb="$(pwd)/comb"

if [ ! -x "${run_comb}" ]; then
   echo "comb executable not found at ${run_comb}"
   exit 1
fi

# Choose arguments for comb
# elements on one side of the cube for each process
elems_per_procs_per_side=100 # 50 100 200
# overall size of the grid
let size=elems_per_procs_per_side
comb_args="${size}_${size}_${size}"
# divide the grid into a number of procs per side
comb_args="${comb_args} -divide 1_1_1"
# set the grid to be periodic in each dimension
comb_args="${comb_args} -periodic 1_1_1"
# set the halo width or number of ghost zones
comb_args="${comb_args} -ghost 1_1_1"
# set number of grid variables
comb_args="${comb_args} -vars 3"
# set number of communication cycles
comb_args="${comb_args} -cycles 25" # 100
# set cutoff between large and small message packing/unpacking kernels
comb_args="${comb_args} -comm cutoff 250"
# set the number of omp threads per process
comb_args="${comb_args} -omp_threads 1"
# disable seq execution tests
comb_args="${comb_args} -exec enable seq"
# disable host memory tests
comb_args="${comb_args} -memory enable host"
# enable mock communication tests
comb_args="${comb_args} -comm enable mock"
# disable mpi communication tests
comb_args="${comb_args} -comm disable mpi"
# disable fusing packs per variable per message
comb_args="${comb_args} -comm disallow per_message_pack_fusing"


# set up arguments for communication method
wait_any_method="-comm post_recv wait_all -comm post_send wait_all -comm wait_recv wait_all -comm wait_send wait_all"

# set up the base command to run a test
# use sep_out.bash to separate each rank's output
run_test_base="${run_comb}"

# for each communication method
for comm_method in "${wait_any_method}"; do

   # Run a test with this comm method
   echo "${run_test_base} ${comm_method} ${comb_args}"
   ${run_test_base} ${comm_method} ${comb_args}

done

echo "done"
