#!/bin/bash

nodes=$1
procs=$2
procs_per_side=$3

# extra arguments to comb (always starts with a space or is empty)
comb_xargs=""

# Choose a command to run mpi based on the system being used
if [[ ! "x" == "x$SYS_TYPE" ]]; then
   if [[ "x$SYS_TYPE" =~ xblueos.*_p9 ]]; then
      # Command used to run mpi on sierra systems
      run_mpi="lrun -N$nodes -p$procs"
      # add arguments to turn on cuda aware mpi (optionally disable gpu direct)
      # run_mpi="${run_mpi} --smpiargs \"-gpu\""
      # run_mpi="${run_mpi} --smpiargs \"-gpu -disable_gdr\""
      # comb_xargs="${comb_xargs} -cuda_aware_mpi"
   elif [[ "x$SYS_TYPE" =~ xblueos.* ]]; then
      # Command used to run mpi on EA systems
      run_mpi="mpirun -np $procs /usr/tcetmp/bin/mpibind"
   else
      # Command used to run mpi on slurm scheduled systems
      run_mpi="srun -N$nodes -n$procs"
   fi
else
   # Command used to run mpi with mpirun
   # https://www.open-mpi.org/doc/v2.0/man1/mpirun.1.php
   # Note: you may need to use additional options to get reasonable mpi behavior
   # --host=hostname0,hostname1,... https://www.open-mpi.org/faq/?category=running#mpirun-hostfile
   # --hostfile my_hosts            https://www.open-mpi.org/faq/?category=running#mpirun-host
   run_mpi="mpirun -np $procs"

   # Command used to run mpi with mpiexec
   # https://www.mpich.org/static/docs/v3.1/www1/mpiexec.html
   # run_mpi="mpiexec -n $procs"
fi

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

# Choose arguments for comb
# overall size of the grid
let size=procs_per_side*100
comb_args="${size}_${size}_${size}"
# divide the grid into a number of procs per side
comb_args="${comb_args} -divide ${procs_per_side}_${procs_per_side}_${procs_per_side}"
# set the grid to be periodic in each dimension
comb_args="${comb_args} -periodic 1_1_1"
# set the halo width or number of ghost zones
comb_args="${comb_args} -ghost 1_1_1"
# set number of grid variables
comb_args="${comb_args} -vars 3"
# set number of communication cycles
comb_args="${comb_args} -cycles 100"
# set cutoff between large and small message packing/unpacking kernels
comb_args="${comb_args} -comm cutoff 250"
# set the number of omp threads per process
comb_args="${comb_args} -omp_threads 10"
# enable all execution tests
comb_args="${comb_args} -exec enable all"
# enable all memory tests
comb_args="${comb_args} -memory enable all"
# enable all communication tests
comb_args="${comb_args} -comm enable all"
# disable mpi_type execution tests (MPI Packing)
# comb_args="${comb_args} -exec disable mpi_type"

# add extra arguments for features enabled outside of the comb args block
comb_args="${comb_args}${comb_xargs}"

# set up arguments for a variety of communication methods
wait_all_method="-comm post_recv wait_all -comm post_send wait_all -comm wait_recv wait_all -comm wait_send wait_all"
wait_some_method="-comm post_recv wait_some -comm post_send wait_some -comm wait_recv wait_some -comm wait_send wait_some"
wait_any_method="-comm post_recv wait_any -comm post_send wait_any -comm wait_recv wait_any -comm wait_send wait_any"

test_all_method="-comm post_recv wait_all -comm post_send test_all -comm wait_recv wait_all -comm wait_send wait_all"
test_some_method="-comm post_recv wait_some -comm post_send test_some -comm wait_recv wait_some -comm wait_send wait_some"
test_any_method="-comm post_recv wait_any -comm post_send test_any -comm wait_recv wait_any -comm wait_send wait_any"

# set up the base command to run a test
# use sep_out.bash to separate each rank's output
run_test_base="${run_mpi} ${run_comb}"

# for each communication method
for comm_method in "${wait_all_method}" "${wait_some_method}" "${wait_any_method}" "${test_all_method}" "${test_some_method}" "${test_any_method}"; do

   # Run a test with this comm method
   echo "${run_test_base} ${comm_method} ${comb_args}"
   ${run_test_base} ${comm_method} ${comb_args}

done

echo "done"
