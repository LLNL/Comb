# Comb v0.1.0

Comb is a communication performance benchmarking tool. It is used to determine performance tradeoffs in implementing communication patterns on high performance computing (HPC) platforms. At its core comb runs combinations of communication patterns with execution patterns, and memory spaces in order to find efficient combinations. The current set of capabilities Comb provides includes:
  - Configurable structured mesh halo exchange communication.
  - A variety of communication patterns based on grouping messages.
  - A variety of execution patterns including serial, openmp threading, cuda, cuda batched kernels, and cuda persistent kernels.
  - A variety of memory spaces including default system allocated memory, pinned host memory, cuda device memory, and cuda managed memory with different cuda memory advice.

It is important to note that Comb is very much a work-in-progress. Additional features will appear in future releases.


## Quick Start

The Comb code lives in a GitHub [repository](https://github.com/llnl/comb). To clone the repo, use the command:

    git clone --recursive https://github.com/llnl/comb.git

On an lc system you can build Comb using the provided cmake scripts and host-configs.

    ./scripts/lc-builds/blueos/nvcc_9_2_gcc_4_9_3.sh
    cd build_lc_blueos_nvcc_9_2_gcc_4_9_3
    make

You can also create your own script and host-config provided you have a C++ compiler that supports the C++11 standard, an MPI library with compiler wrapper, and optionally an install of cuda 9.0 or later.

    ./scripts/my-builds/compiler_version.sh
    cd build_my_compiler_version
    make


## User Documentation

Minimal documentation is available.

Comb runs every combination of execution pattern, and memory space enabled. Each rank prints its results to stdout. The [sep_out.bash](./scripts/sep_out.bash) script may be used to simplify data collection by piping the output of each rank into a different file. The [combine_output.lua](./scripts/combine_output.lua) lua script may be used to simplify data aggregation from multiple files.

### Configure Options

The cmake configuration options change which execution patterns and memory spaces are enabled.

  - __ENABLE_OPENMP__ Allow use of openmp and enable test combinations using openmp
  - __ENABLE_CUDA__  Allow use of cuda and enable test combinations using cuda

### Runtime Options

The runtime options change the properties of the grid and its decomposition, as well as the communication pattern used.

  -   __\#\_\#\_\#__ Grid size in each dimension (Required)
  -   __\-divide \#\_\#\_\#__ Number of subgrids in each dimension (Required)
  -   __\-periodic \#\_\#\_\#__ Periodicity in each dimension
  -   __\-ghost \#__ The halo width or number of ghost zones
  -   __\-vars \#__ The number of grid variables
  -   __\-comm *option*__ Communication options
      -   __mock__ Do mock communication (do not make MPI calls)
      -   __cutoff \#__ Number of elements cutoff between large and small message packing kernels
      -   __post_recv *option*__ Communication post receive (MPI_Irecv) options
          -   __wait_any__ Post recvs one-by-one
          -   __wait_some__ Post recvs one-by-one
          -   __wait_all__ Post recvs one-by-one
          -   __test_any__ Post recvs one-by-one
          -   __test_some__ Post recvs one-by-one
          -   __test_all__ Post recvs one-by-one
      -   __post_send *option*__ Communication post send (MPI_Isend) options
          -   __wait_any__ pack and send messages one-by-one
          -   __wait_some__ pack messages then send them in groups
          -   __wait_all__ pack all messages then send them all
          -   __test_any__ pack messages asynchronously and send when ready
          -   __test_some__ pack multiple messages asynchronously and send when ready
          -   __test_all__ pack all messages asynchronously and send when ready
      -   __wait_recv *option*__ Communication wait to recv and unpack (MPI_Wait, MPI_Test) options
          -   __wait_any__ recv and unpack messages one-by-one (MPI_Waitany)
          -   __wait_some__ recv messages then unpack them in groups (MPI_Waitsome)
          -   __wait_all__ recv all messages then unpack them all (MPI_Waitall)
          -   __test_any__ recv and unpack messages one-by-one (MPI_Testany)
          -   __test_some__ recv messages then unpack them in groups (MPI_Testsome)
          -   __test_all__ recv all messages then unpack them all (MPI_Testall)
      -   __wait_send *option*__ Communication wait on sends (MPI_Wait, MPI_Test) options
          -   __wait_any__ Wait for each send to complete one-by-one (MPI_Waitany)
          -   __wait_some__ Wait for all sends to complete in groups (MPI_Waitsome)
          -   __wait_all__ Wait for all sends to complete (MPI_Waitall)
          -   __test_any__ Wait for each send to complete one-by-one by polling (MPI_Testany)
          -   __test_some__ Wait for all sends to complete in groups by polling (MPI_Testsome)
          -   __test_all__ Wait for all sends to complete by polling (MPI_Testall)
  -   __\-cycles \#__ Number of times the communication pattern is tested
  -   __\-omp_threads \#__ Number of openmp threads requested

#### Example Script

The [run_scale_tests.bash](./scripts/run_scale_tests.bash) is an example script that allocates resources and runs the code in a variety of configurations via the scale_tests.bash script. The run_scale_tests.bash script takes a single argument, the number of processes per side used to split the grid into an N x N x N decomposition.

    mkdir 1_1_1
    cd 1_1_1
    ln -s path/to/comb/build/bin/comb .
    ln -s path/to/comb/scripts/* .
    ./run_scale_tests.bash 1

The [scale_tests.bash](./scripts/scale_tests.bash) script used run_scale_tests.bash which shows the options available and how the code may be run with mpi.


## Related Software

The [**RAJA Performance Suite**](https://github.com/LLNL/RAJAPerf) contains a collection of loop kernels implemented in multiple RAJA and non-RAJA variants. We use it to monitor and assess RAJA performance on different platforms using a variety of compilers.

The [**RAJA Proxies**](https://github.com/LLNL/RAJAProxies) repository contains RAJA versions of several important HPC proxy applications.


## Contributions

The Comb team follows the [GitFlow](http://nvie.com/posts/a-successful-git-branching-model/) development model. Folks wishing to contribute to Comb, should include their work in a feature branch created from the Comb `develop` branch. Then, create a pull request with the `develop` branch as the destination. That branch contains the latest work in Comb. Periodically, we will merge the develop branch into the `master` branch and tag a new release.


## Authors

Thanks to all of Comb's
[contributors](https://github.com/LLNL/Comb/graphs/contributors).

Comb was created by Jason Burmark (burmark1@llnl.gov).


## Release

Comb is released under an MIT license. For more details, please see the
[LICENSE](./LICENSE), [RELEASE](./RELEASE), and [NOTICE](./NOTICE) files.

`LLNL-CODE-758885`
