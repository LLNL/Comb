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

You can build Comb using the provided Makefile, provided you have a C++ compiler that supports the C++11 standard, an MPI library with compiler wrapper for said C++ compiler, and an install of cuda 9.0 or later.
Simply change the CUDA_COMPILER and CXX_MPI_COMPILER in the Makefile to those on your system.

    make

User Documentation
-------------------

Minimal documentation is available.

The [scale_tests.bash](./scripts/scale_tests.bash) script shows the basic options available and how the code may be run.

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
