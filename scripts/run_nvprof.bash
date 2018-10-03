#!/bin/bash

# runs nvprof on each rank separately creating a .nvprof output file
# separates the output of each mpi rank into a different file

ARGS="$@"
ARGS_UNDERSCORE="$(sed s/\ /_/g <<<$ARGS)"
ARGS_UNDERSCORE="$(sed s-/-@-g <<<$ARGS_UNDERSCORE)"
ARGS_UNDERSCORE="$(echo $ARGS_UNDERSCORE | cut -c -192)"

# find the environment variable with the mpi rank of this process
if [[ ! "x" == "x$JSM_NAMESPACE_RANK" ]]; then
   RANK=${JSM_NAMESPACE_RANK}
	RANK_VAR="JSM_NAMESPACE_RANK"
elif [[ ! "x" == "x$OMPI_COMM_WORLD_RANK" ]]; then
   RANK=${OMPI_COMM_WORLD_RANK}
	RANK_VAR="OMPI_COMM_WORLD_RANK"
elif [[ ! "x" == "x$MPIRUN_RANK" ]]; then
   RANK=${MPIRUN_RANK}
	RANK_VAR="MPIRUN_RANK"
else
	echo "Could not find mpirank" 1>&2
	exit 1
fi

# attempt to find the name of the node this mpi rank is running on
if [[ ! "x" == "x$nodename" ]]; then
	NODE="$nodename"
	NODE_VAR="nodename"
elif [[ ! "x" == "x$SLURMD_NODENAME" ]]; then
	NODE="$SLURMD_NODENAME"
	NODE_VAR="SLURMD_NODENAME"
elif [[ ! "x" == "x$LCSCHEDCLUSTER" ]]; then
	NODE="$LCSCHEDCLUSTER"
	NODE_VAR="LCSCHEDCLUSTER"
fi

# create an identifier for this process using its rank
PROC_NAME="${RANK}"
PROC_NAME_VAR="%q{${RANK_VAR}}"

# add the nodename to the process identifier if available
if [[ ! "x" == "x$NODE_VAR" ]]; then
	PROC_NAME="${PROC_NAME}_${NODE}"
	PROC_NAME_VAR="{PROC_NAME_VAR}_(%q{${NODE_VAR}})"
fi

# use args and rank to make file name
OUT_FILE_NVPROF="runnvprof.${ARGS_UNDERSCORE}.${PROC_NAME_VAR}"
OUT_FILE="runnvprof.${ARGS_UNDERSCORE}.${PROC_NAME}"
if [ -f "$OUT_FILE" ]; then
	echo "File already exists $OUT_FILE" 1>&2
	exit 1
fi

# options to pass to nvprof
NVPROF_OPTS="-o ${OUT_FILE_NVPROF}.nvprof"
# NVPROF_OPTS="$NVPROF_OPTS --profile-from-start off"
# NVPROF_OPTS="$NVPROF_OPTS -f"
# NVPROF_OPTS="$NVPROF_OPTS --process-name \"MPI Rank ${PROC_NAME_VAR}\""
# NVPROF_OPTS="$NVPROF_OPTS --system-profiling on"
# NVPROF_OPTS="$NVPROF_OPTS --demangling on"
# NVPROF_OPTS="$NVPROF_OPTS --cpu-profiling off"
# NVPROF_OPTS="$NVPROF_OPTS --unified-memory-profiling per-process-device"
# NVPROF_OPTS="$NVPROF_OPTS --cpu-thread-tracing on"

# find nvprof
NVPROF="$(which nvprof)"
if [ ! -f "$NVPROF" ]; then
	echo "Could not find $NVPROF" 1>&2
	exit 1
fi
NVPROF="$NVPROF $NVPROF_OPTS"

# print the command to be executed for the 0th rank
if [[ "x0" == "x$RANK" ]]; then
	echo "$NVPROF $ARGS &> ${OUT_FILE}"
fi
# execute nvprof and the executable and redirect its output to a file
exec $NVPROF $ARGS &> "${OUT_FILE}"
