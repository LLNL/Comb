# Comb v0.1.2

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

To run basic tests make a directory and make symlinks to the comb executable and scripts. The scripts expect a symlink to comb to exist in the run directory. The run_tests.bash script runs the basic_tests.bash script in 2^3 processes.

    ln -s /path/to/comb/build_my_compiler_version/bin/comb .
    ln -s /path/to/comb/scripts/* .
    ./run_tests.bash 2 basic_tests.bash

## User Documentation

Minimal documentation is available.

Comb runs every combination of execution pattern, and memory space enabled. Each rank prints its results to stdout. The [sep_out.bash](./scripts/sep_out.bash) script may be used to simplify data collection by piping the output of each rank into a different file. The [combine_output.lua](./scripts/combine_output.lua) lua script may be used to simplify data aggregation from multiple files.

Comb uses a variety of manual packing/unpacking execution techniques such as sequential, openmp, and cuda. Comb also uses MPI_Pack/MPI_Unpack with MPI derived datatypes for packing/unpacking. (Note: tests using cuda managed memory and MPI datatypes are disabled as they sometimes produce incorrect results)

Comb creates a different MPI communicator for each test. This communicator is assigned a generic name unless MPI datatypes are used for packing and unpacking. When MPI datatypes are used the name of the memory allocator is appended to the communicator name.

### Configure Options

The cmake configuration options change which execution patterns and memory spaces are enabled.

  - __ENABLE_MPI__ Allow use of mpi and enable test combinations using mpi
  - __ENABLE_OPENMP__ Allow use of openmp and enable test combinations using openmp
  - __ENABLE_CUDA__  Allow use of cuda and enable test combinations using cuda

### Runtime Options

The runtime options change the properties of the grid and its decomposition, as well as the communication pattern used.

  -   __*\#\_\#\_\#*__ Grid size in each dimension (Required)
  -   __\-divide *\#\_\#\_\#*__ Number of subgrids in each dimension (Required)
  -   __\-periodic *\#\_\#\_\#*__ Periodicity in each dimension
  -   __\-ghost *\#\_\#\_\#*__ The halo width or number of ghost zones in each dimension
  -   __\-vars *\#*__ The number of grid variables
  -   __\-comm *option*__ Communication options
      -   __cutoff *\#*__ Number of elements cutoff between large and small message packing kernels
      -   __enable|disable *option*__ Enable or disable specific message passing execution policies
          -   __all__ all message passing execution patterns
          -   __mock__ mock message passing execution pattern (do not communicate)
          -   __mpi__ mpi message passing execution pattern
          -   __gpump__ libgpump message passing execution pattern
          -   __mp__ libmp message passing execution pattern (experimental)
          -   __umr__ umr message passing execution pattern (experimental)
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
      -   __allow|disallow *option*__ Allow or disallow specific communications options
          -   __per_message_pack_fusing__ Allow packing kernels to be fused for a single variable when packing into the same message
  -   __\-cycles *\#*__ Number of times the communication pattern is tested
  -   __\-omp_threads *\#*__ Number of openmp threads requested
  -   __\-exec *option*__ Execution options
      -   __enable|disable *option*__ Enable or disable specific execution patterns
          -   __all__ all execution patterns
          -   __seq__ sequential CPU execution pattern
          -   __omp__ openmp threaded CPU execution pattern
          -   __cuda__ cuda GPU execution pattern
          -   __cuda_graph__ cuda GPU batched via cuda graph API execution pattern
          -   __cuda_batch__ cuda GPU batched kernel execution pattern
          -   __cuda_batch_fewgs__ cuda GPU batched kernel with few grid synchronizations execution pattern
          -   __cuda_persistent__ cuda GPU persistent kernel execution pattern
          -   __cuda_persistent_fewgs__ cuda GPU persistent kernel with few grid synchronizations execution pattern
          -   __mpi_type__ MPI datatypes MPI implementation execution pattern
  -   __\-memory *option*__ Memory space options
      -   __enable|disable *option*__ Enable or disable specific memory spaces for mesh allocations
          -   __all__ all memory spaces
          -   __host__ host CPU memory space
          -   __cuda_pinned__ cuda pinned memory space
          -   __cuda_device__ cuda device memory space
          -   __cuda_managed__ cuda managed memory space
          -   __cuda_managed_host_preferred__ cuda managed with host preferred advice memory space
          -   __cuda_managed_host_preferred_device_accessed__ cuda managed with host preferred and device accessed advice memory space
          -   __cuda_managed_device_preferred__ cuda managed with device preferred advice memory space
          -   __cuda_managed_device_preferred_host_accessed__ cuda managed with device preferred and host accessed advice memory space
  -   __\-cuda_aware_mpi__ Assert that you are using a cuda aware mpi implementation and enable tests that pass cuda device or managed memory to MPI
  -   __\-cuda_host_accessible_from_device__ Assert that your system supports pageable host memory access from the device and enable tests that access pageable host memory on the device

### Example Script

The [run_tests.bash](./scripts/run_tests.bash) is an example script that allocates resources and uses a script such as [focused_tests.bash](./scripts/focused_tests.bash) to run the code in a variety of configurations. The run_tests.bash script takes two arguments, the number of processes per side used to split the grid into an N x N x N decomposition, and the tests script.

    mkdir 1_1_1
    cd 1_1_1
    ln -s path/to/comb/build/bin/comb .
    ln -s path/to/comb/scripts/* .
    ./run_tests.bash 1 focused_tests.bash

The [scale_tests.bash](./scripts/scale_tests.bash) script used with run_tests.bash which shows the options available and how the code may be run with multiple sets of arguments with mpi.
The [focused_tests.bash](./scripts/focused_tests.bash) script used with run_tests.bash which shows the options available and how the code may be run with one set of arguments with mpi.

### Output

Comb outputs Comb\_(number)\_summary and Comb\_(number)\_proc(number) files. The summary file contains aggregated results from the proc files which contain per process results.
The files contain the argument and code setup information and the results of multiple tests. The results for each test follow a line started with "Starting test" and the name of the test.

The first set of tests are memory copy tests with names of the following form.

    Starting test memcpy (execution policy) dst (destination memory space) src (source memory space)"
    copy_sync-(number of variables)-(elements per variable)-(bytes per element): num (number of repeats) avg (time) s min (time) s max (time) s
Example:

    Starting test memcpy seq dst Host src Host
    copy_sync-3-1061208-8: num 200 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
This is a test in which memory is copied via sequential cpu execution to one host memory buffer from another host memory buffer.
The test involves one measurement.

    copy_sync-3-1061208-8 Copying 3 buffers of 1061208 elements of size 8.

The second set of tests are the message passing tests with names of the following form.

    Comm (message passing execution policy) Mesh (physics execution policy) (mesh memory space) Buffers (large message execution policy) (large message memory space) (small message execution policy) (small message memory space)
    (test phase): num (number of repeats) avg (time) s min (time) s max (time) s
    ...
Example

    Comm mpi Mesh seq Host Buffers seq Host seq Host
    pre-comm:  num 200 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
    post-recv: num 200 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
    post-send: num 200 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
    wait-recv: num 200 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
    wait-send: num 200 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
    post-comm: num 200 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
    start-up:   num 8 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
    test-comm:  num 8 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
    bench-comm: num 8 avg 0.123456789 s min 0.123456789 s max 0.123456789 s
This is a test in which a mesh is updated with physics running via sequential cpu execution using memory allocated in host memory. The buffers used for large messages are packed/unpacked via sequential cpu execution and allocated in host memory and the buffers used with MPI for small messages are packed/unpacked via sequential cpu execution and allocated in host memory.
This test involves multiple measurements, the first six time individual parts of the physics cycle and communication.
  - pre-comm "Physics" before point-to-point communication, in this case setting memory to initial values.
  - post-recv Allocating MPI receive buffers and calling MPI_Irecv.
  - post-send Allocating MPI send buffers, packing buffers, and calling MPI_Isend.
  - wait-recv Waiting to receive MPI messages, unpacking MPI buffers, and freeing MPI receive buffers
  - wait-send Waiting for MPI send messages to complete and freeing MPI send buffers.
  - post-comm "Physics" after point-to-point communication, in this case resetting memory to initial values.
The final three measure problem setup, correctness testing, and total benchmark time.
  - start-up Setting up mesh and point-to-point communication.
  - test-comm Testing correctness of point-to-point communication.
  - bench-comm Running benchmark, starts after an initial MPI_Barrier and ends after a final MPI_Barrier.

##### Execution Policies

  - __seq__ Sequential CPU execution
  - __omp__ Parallel CPU execution via OpenMP
  - __cuda__ Parallel GPU execution via cuda
  - __cudaGraph__ Parallel GPU execution via cuda graphs
  - __cudaBatch__ Parallel GPU execution via kernel batching
  - __cudaBatch_fewgs__ Parallel GPU execution via kernel batching without grid synchronization between kernels
  - __cudaPersistent__ Parallel GPU execution via persistent kernel batching
  - __cudaPersistent_fewgs__ Parallel GPU execution via persistent kernel batching without grid synchronization between kernels
  - __mpi_type__ Packing or unpacking execution done via mpi datatypes used with MPI_Pack/MPI_Unpack

##### Memory Spaces

  - __Host__ CPU memory (malloc)
  - __HostPinned__ Cuda Pinned CPU memory (cudaHostAlloc)
  - __Device__ Cuda GPU memory (cudaMalloc)
  - __Managed__ Cuda Managed GPU memory (cudaMallocManaged)
  - __ManagedHostPreferred__ Cuda Managed CPU Pinned memory (cudaMallocManaged + cudaMemAdviseSetPreferredLocation cudaCpuDeviceId)
  - __ManagedHostPreferredDeviceAccessed__ Cuda Managed CPU Pinned memory (cudaMallocManaged + cudaMemAdviseSetPreferredLocation cudaCpuDeviceId + cudaMemAdviseSetAccessedBy 0)
  - __ManagedDevicePreferred__ Cuda Managed CPU Pinned memory (cudaMallocManaged + cudaMemAdviseSetPreferredLocation 0)
  - __ManagedDevicePreferredHostAccessed__ Cuda Managed CPU Pinned memory (cudaMallocManaged + cudaMemAdviseSetPreferredLocation 0 + cudaMemAdviseSetAccessedBy cudaCpuDeviceId)

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
