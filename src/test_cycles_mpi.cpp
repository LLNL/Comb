//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-758885
//
// All rights reserved.
//
// This file is part of Comb.
//
// For details, see https://github.com/LLNL/Comb
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#include "comb.hpp"

#ifdef COMB_ENABLE_MPI

#include "comm_pol_mpi.hpp"
#include "do_cycles.hpp"

namespace COMB {

void test_cycles_mpi(CommInfo& comminfo, MeshInfo& info,
                     COMB::Executors& exec,
                     COMB::Allocators& alloc,
                     IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  CommContext<mpi_pol> con_comm{exec.base_mpi.get()};

  {
    // mpi host memory tests
    AllocatorInfo& cpu_many_aloc = alloc.host;
    AllocatorInfo& cpu_few_aloc  = alloc.host;

  #ifdef COMB_ENABLE_CUDA
    AllocatorInfo& cuda_many_aloc = alloc.cuda_hostpinned;
    AllocatorInfo& cuda_few_aloc  = alloc.cuda_hostpinned;
  #else
    AllocatorInfo& cuda_many_aloc = alloc.invalid;
    AllocatorInfo& cuda_few_aloc  = alloc.invalid;
  #endif

    do_cycles_allocators(con_comm,
                         comminfo, info,
                         exec,
                         alloc,
                         cpu_many_aloc, cpu_few_aloc,
                         cuda_many_aloc, cuda_few_aloc,
                         num_vars, ncycles, tm, tm_total);
  }

#ifdef COMB_ENABLE_CUDA
  {
    // mpi cuda memory tests
    AllocatorInfo& cpu_many_aloc = alloc.cuda_device;
    AllocatorInfo& cpu_few_aloc  = alloc.cuda_device;

    AllocatorInfo& cuda_many_aloc = alloc.cuda_device;
    AllocatorInfo& cuda_few_aloc  = alloc.cuda_device;

    do_cycles_allocators(con_comm,
                         comminfo, info,
                         exec,
                         alloc,
                         cpu_many_aloc, cpu_few_aloc,
                         cuda_many_aloc, cuda_few_aloc,
                         num_vars, ncycles, tm, tm_total);
  }
#endif

}

} // namespace COMB

#endif
