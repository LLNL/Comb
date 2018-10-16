Comb v0.1.0
============

Comb is a communication performance benchmarking tool. It is used to determine performance tradeoffs in implementing communication patterns on high performance computing (HPC) platforms. At its core comb runs combinations of communication patterns with execution patterns, and memory spaces. The current set of capabilities Comb provides includes:
  * Configurable structured mesh halo exchange communication.
  * A variety of communication patterns based on grouping messages.
  * A variety of execution patterns including serial, openmp threading, cuda, cuda batched kernels, and cuda persistent kernels.
  * A variety of memory spaces including default system allocated memory, pinned host memory, cuda device memory, and cuda managed memory with different cuda memory advice.

It is important to note that Comb is very much a work-in-progress. Additional features will appear in future releases.

Quick Start
-----------

The Comb code lives in a GitHub [repository](https://github.com/llnl/comb). To clone the repo, use the command:

    git clone --recursive https://github.com/llnl/comb.git

You can build Comb using the provided cmake scripts and host-configs if you are using an lc system.

    ./scripts/lc-builds/blueos/nvcc_9_2_gcc_4_9_3.sh
    cd build_lc_blueos_nvcc_9_2_gcc_4_9_3
    make

You can also create your own script and host-config provided you have a C++ compiler that supports the C++11 standard, an MPI library with compiler wrapper for said C++ compiler, and optionally an install of cuda 9.0 or later.

    ./scripts/my-builds/compiler_version.sh
    cd build_my_compiler_version
    make

User Documentation
-------------------

Minimal documentation is available.

The [run_scale_tests.bash](./scripts/run_scale_tests.bash) is an example script that allocates resources and runs the code in a variety of configurations via the scale_tests.bash script. The run_scale_tests.bash script takes a single argument, the number of processes per side used to split the grid into an N x N x N decomposition.
    * `mkdir 1_1_1`
    * `ln -s path/to/comb/comb comb`
    * `ln -s path/to/comb/scripts/* .`
    * `./run_scale_tests.bash 1`
The [scale_tests.bash](./scripts/scale_tests.bash) script used run_scale_tests.bash which shows the options available and how the code may be run with mpi.


Related Software
--------------------

The [**RAJA Performance Suite**](https://github.com/LLNL/RAJAPerf) contains a collection of loop kernels implemented in multiple RAJA and non-RAJA variants. We use it to monitor and assess RAJA performance on different platforms using a variety of compilers.

The [**RAJA Proxies**](https://github.com/LLNL/RAJAProxies) repository contains RAJA versions of several important HPC proxy applications.

Contributions
---------------

The Comb team follows the [GitFlow](http://nvie.com/posts/a-successful-git-branching-model/) development model. Folks wishing to contribute to Comb, should include their work in a feature branch created from the Comb `develop` branch. Then, create a pull request with the `develop` branch as the destination. That branch contains the latest work in Comb. Periodically, we will merge the develop branch into the `master` branch and tag a new release.

Authors
-----------

Thanks to all of Comb's
[contributors](https://github.com/LLNL/Comb/graphs/contributors).

Comb was created by Jason Burmark (burmark1@llnl.gov).


Release
-----------

Comb is released under an MIT license. For more details, please see the
[LICENSE](./LICENSE), [RELEASE](./RELEASE), and [NOTICE](./NOTICE) files.

`LLNL-CODE-758885`
