//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2021, Lawrence Livermore National Security, LLC.
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

#ifdef COMB_ENABLE_MP

#include "comm_pol_mp.hpp"
#include "do_cycles_allocators.hpp"

namespace COMB {

void test_cycles_mp(CommInfo& comminfo, MeshInfo& info,
                       COMB::Executors& exec,
                       COMB::Allocators& alloc,
                       IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  CommContext<mp_pol> con_comm{exec.base_cuda.get()};

#ifdef COMB_ENABLE_CUDA
  AllocatorInfo& cpu_many_aloc = alloc.cuda_device;
  AllocatorInfo& cpu_few_aloc  = alloc.cuda_device;

  AllocatorInfo& gpu_many_aloc = alloc.cuda_device;
  AllocatorInfo& gpu_few_aloc  = alloc.cuda_device;
#else
  AllocatorInfo& cpu_many_aloc = alloc.invalid;
  AllocatorInfo& cpu_few_aloc  = alloc.invalid;

  AllocatorInfo& gpu_many_aloc = alloc.invalid;
  AllocatorInfo& gpu_few_aloc  = alloc.invalid;
#endif

  do_cycles_allocators(con_comm,
                       comminfo, info,
                       exec,
                       alloc,
                       cpu_many_aloc, cpu_few_aloc,
                       gpu_many_aloc, gpu_few_aloc,
                       num_vars, ncycles, tm, tm_total);

}

} // namespace COMB

#endif // COMB_ENABLE_MP
