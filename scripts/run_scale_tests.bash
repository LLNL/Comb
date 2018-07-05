#!/bin/bash

procs_per_side=$1

procs_per_node=4

let procs=procs_per_side*procs_per_side*procs_per_side
let nodes=(procs+procs_per_node-1)/procs_per_node

get_nodes="bsub -nnodes ${nodes} -core_isolation 2 -W 60 -G guests -q pdebug -Is"
run_tests="scale_tests.bash $nodes $procs $procs_per_side"

full_test="${get_nodes} ${run_tests}"

echo "${full_test}"
time ${full_test}

