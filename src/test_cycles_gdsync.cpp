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

#ifdef COMB_ENABLE_GDSYNC

#include "comm_pol_gdsync.hpp"
#include "do_cycles.hpp"

namespace COMB {

void test_cycles_gdsync(CommInfo& comminfo, MeshInfo& info,
                        COMB::ExecContexts& exec,
                        COMB::Allocators& alloc,
                        COMB::ExecutorsAvailable& exec_avail,
                        IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  CommContext<gdsync_pol> con_comm{exec.base_cuda};

#ifdef COMB_ENABLE_CUDA
  AllocatorInfo& cpu_many_aloc = alloc.cuda_device;
  AllocatorInfo& cpu_few_aloc  = alloc.cuda_device;

  AllocatorInfo& cuda_many_aloc = alloc.cuda_device;
  AllocatorInfo& cuda_few_aloc  = alloc.cuda_device;
#else
  AllocatorInfo& cpu_many_aloc = alloc.invalid;
  AllocatorInfo& cpu_few_aloc  = alloc.invalid;

  AllocatorInfo& cuda_many_aloc = alloc.invalid;
  AllocatorInfo& cuda_few_aloc  = alloc.invalid;
#endif

  do_cycles_allocators(con_comm,
                       comminfo, info,
                       exec,
                       alloc,
                       cpu_many_aloc, cpu_few_aloc,
                       cuda_many_aloc, cuda_few_aloc,
                       exec_avail,
                       num_vars, ncycles, tm, tm_total);

}

} // namespace COMB

#endif // COMB_ENABLE_GDSYNC
