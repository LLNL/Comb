#!/bin/bash

procs_per_side=$1
let procs=procs_per_side*procs_per_side*procs_per_side

# Choose a command to get nodes
if [[ ! "x" == "x$SYS_TYPE" ]]; then
   if [[ "x$SYS_TYPE" =~ xblueos.*_p9 ]]; then
      # Command used to get nodes on sierra systems

      procs_per_node=4
      let nodes=(procs+procs_per_node-1)/procs_per_node

      # get_nodes="bsub -nnodes ${nodes} -core_isolation 2 -W 60 -G guests -Is -XF"
      get_nodes="lalloc ${nodes} -W 60 --shared-launch"

   elif [[ "x$SYS_TYPE" =~ xblueos.* ]]; then
      # Command used to get nodes on EA systems

      procs_per_node=4
      let nodes=(procs+procs_per_node-1)/procs_per_node

      get_nodes="bsub -n ${procs} -R \"span[ptile=${procs_per_node}]\" -W 60 -G guests -Is -XF"

   else
      # Command used to get nodes on slurm scheduled systems

      procs_per_node=1
      let nodes=(procs+procs_per_node-1)/procs_per_node

      get_nodes="salloc -N${nodes}"

   fi
else
   # Command used to get nodes on other systems
   procs_per_node=1
   let nodes=(procs+procs_per_node-1)/procs_per_node

   # Don't know how to get nodes, defer to mpi in next script
   get_nodes=""

fi

run_tests="scale_tests.bash $nodes $procs $procs_per_side"

full_test="${get_nodes} ${run_tests}"

echo "${full_test}"
time ${full_test}
