#!/bin/bash

procs_per_node=-1
procs_per_side=""
test_script=""

positional_arg=0

################################################################################
#
# Usage:
#     run_tests.bash [args to run_tests.bash] procs_per_side test_script
#
# Parse any args for this script beginning with - and consume them using shift
# leave the program to profile, if any, and its args
#
# Examples:
#     run_tests.bash 2 focused_tests.bash
#       # Launch focused_tests.bash with 2x2x2 procs with default procs per node
#
#     run_rocprof -gui [optional rocprof profile file]
#       # run the rocprof gui (only available on x86 machines currently)
#       #   and optionally view the given profile
#
################################################################################
while [ "$#" -gt 0 ]; do

   if [[ "$1" =~ ^\-.* ]]; then

      if [[ "x$1" == "x-ppn" || "x$1" == "x--procs-per-node" ]]; then

         if [ "$#" -le 1 ]; then
            echo "missing argument to $1" 1>&2
            exit 1
         fi

         natural_re='^[0-9]+$'
         if ! [[ "$2" =~ $natural_re ]]; then
            echo "invalid arguments $1 $2: argument to $1 must be a number" 1>&2
            exit 1
         fi

         procs_per_node="$2"
         shift

      else

         echo "unknown arg $1" 1>&2
         exit 1

      fi

   else

      if [[ "x$positional_arg" == "x0" ]]; then

         procs_per_side="$1"

      elif [[ "x$positional_arg" == "x1" ]]; then

         test_script="$1"

      else

         echo "Found extra positional arg $1" 1>&2
         exit 1

      fi

      let positional_arg=positional_arg+1
   fi

   shift

done

if [[ "x" == "x$procs_per_side" ]]; then
   echo "First positional arg procs_per_side not given" 1>&2
   exit 1
fi
if [[ "x" == "x$test_script" ]]; then
   echo "Second positional arg test_script not given" 1>&2
   exit 1
fi

let procs=procs_per_side*procs_per_side*procs_per_side

if [ ! -f  "$test_script" ]; then
   echo "tests script $test_script not found"
   exit 1
fi

# Choose a command to get nodes
if [[ ! "x" == "x$SYS_TYPE" ]]; then
   if [[ "x$SYS_TYPE" =~ xblueos.*_p9 ]]; then
      # Command used to get nodes on sierra systems

      if [[ "x-1" == "x$procs_per_node" ]]; then
         procs_per_node=4
      fi
      let nodes=(procs+procs_per_node-1)/procs_per_node

      # get_nodes="bsub -nnodes ${nodes} -core_isolation 2 -W 60 -G guests -Is -XF"
      get_nodes="lalloc ${nodes} -W 60 --shared-launch"

   elif [[ "x$SYS_TYPE" =~ xblueos.* ]]; then
      # Command used to get nodes on EA systems

      if [[ "x-1" == "x$procs_per_node" ]]; then
         procs_per_node=4
      fi
      let nodes=(procs+procs_per_node-1)/procs_per_node

      get_nodes="bsub -n ${procs} -R \"span[ptile=${procs_per_node}]\" -W 60 -G guests -Is -XF"

   elif [[ "x$SYS_TYPE" =~ xtoss_4_x86_64_ib_cray ]]; then
      # Command used to get nodes on ElCap EA systems

      if [[ "x-1" == "x$procs_per_node" ]]; then
         procs_per_node=1
      fi
      let nodes=(procs+procs_per_node-1)/procs_per_node

      get_nodes="salloc -N${nodes} --exclusive"

   else
      # Command used to get nodes on slurm scheduled systems

      if [[ "x-1" == "x$procs_per_node" ]]; then
         procs_per_node=1
      fi
      let nodes=(procs+procs_per_node-1)/procs_per_node

      get_nodes="salloc -N${nodes} --exclusive"

   fi
else
   # Command used to get nodes on other systems
   if [[ "x-1" == "x$procs_per_node" ]]; then
      procs_per_node=1
   fi
   let nodes=(procs+procs_per_node-1)/procs_per_node

   # Don't know how to get nodes, defer to mpi in next script
   get_nodes=""

fi

run_tests="$test_script $nodes $procs $procs_per_side"

full_test="${get_nodes} ${run_tests}"

echo "${full_test}"
time ${full_test}
