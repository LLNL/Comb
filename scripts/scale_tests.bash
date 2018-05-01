#!/bin/bash

check_sierra_nodes || exit 1


nodes=$1
procs=$2
procs_per_side=$3

run_nodes="lrun -N$nodes -p$procs"

let size=procs_per_side*100

comb="comb_o"
sizes="${size}_${size}_${size}"
divide="-divide ${procs_per_side}_${procs_per_side}_${procs_per_side}"
periodic="-periodic 1_1_1"
ghost="-ghost 2"
vars="-vars 3"
cycles="-cycles 100"
cutoff="-comm cutoff 250"
omp_threads="-omp_threads 10"
mock="-comm mock"

wait_all_algorithm="-comm post_recv wait_all -comm post_send wait_all -comm wait_recv wait_all -comm wait_send wait_all"
wait_some_algorithm="-comm post_recv wait_all -comm post_send wait_some -comm wait_recv wait_some -comm wait_send wait_all"
wait_any_algorithm="-comm post_recv wait_any -comm post_send wait_any -comm wait_recv wait_any -comm wait_send wait_all"

test_all_algorithm="-comm post_recv wait_all -comm post_send test_all -comm wait_recv wait_all -comm wait_send wait_all"
test_some_algorithm="-comm post_recv wait_all -comm post_send test_some -comm wait_recv wait_some -comm wait_send wait_all"
test_any_algorithm="-comm post_recv wait_any -comm post_send test_any -comm wait_recv wait_any -comm wait_send wait_all"

test_base="${run_nodes} sep_out ${comb} ${sizes} ${divide} ${periodic} ${ghost} ${vars} ${cycles} ${cutoff} ${omp_threads}"

for comm_test in "${test_base} ${wait_all_algorithm}" "${test_base} ${wait_some_algorithm}" "${test_base} ${wait_any_algorithm}" "${test_base} ${test_all_algorithm}" "${test_base} ${test_some_algorithm}" "${test_base} ${test_any_algorithm}"; do
  
  mock_test="${comm_test} ${mock}"
  
  echo "${comm_test}"
  ${comm_test}
  
  echo "${mock_test}"
  ${mock_test}
  
done

echo "done"

