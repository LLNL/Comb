#!/bin/bash

# separates the output of each mpi rank into a different file

ARGS="$@"
ARGS_UNDERSCORE="$(sed s/\ /_/g <<<$ARGS)"
ARGS_UNDERSCORE="$(sed s-/-@-g <<<$ARGS_UNDERSCORE)"
ARGS_UNDERSCORE="$(echo $ARGS_UNDERSCORE | cut -c -192)"

# attempt to find the environment variable with the mpi rank of this process
if [[ ! "x" == "x$JSM_NAMESPACE_RANK" ]]; then
   RANK=${JSM_NAMESPACE_RANK}
elif [[ ! "x" == "x$OMPI_COMM_WORLD_RANK" ]]; then
   RANK=${OMPI_COMM_WORLD_RANK}
elif [[ ! "x" == "x$MPIRUN_RANK" ]]; then
   RANK=${MPIRUN_RANK}
else
   echo "Could not find mpirank" 1>&2
   exit 1
fi

# use args and rank to make file name
OUT_FILE="sepout.${ARGS_UNDERSCORE}.${RANK}"
if [ -f "$OUT_FILE" ]; then
   echo "File already exists $OUT_FILE" 1>&2
   exit 1
fi

# print the command to be executed for the 0th rank
if [[ "x0" == "x$RANK" ]]; then
   echo "$ARGS &> $OUT_FILE"
fi
# execute the executable and redirect its output to a file
exec $ARGS &> $OUT_FILE
