//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

namespace COMB {

template < typename pol >
void do_warmup(ExecContext<pol>& con, COMB::Allocator& aloc, Timer& tm, IdxT num_vars, IdxT len)
{
  tm.clear();

  char test_name[1024] = ""; snprintf(test_name, 1024, "warmup %s %s", pol::get_name(), aloc.name());
  Range r(test_name, Range::green);

  DataT** vars = new DataT*[num_vars];

  for (IdxT i = 0; i < num_vars; ++i) {
    vars[i] = (DataT*)aloc.allocate(len*sizeof(DataT));
  }

  for (IdxT i = 0; i < num_vars; ++i) {

    DataT* data = vars[i];

    con.for_all(0, len, detail::set_n1{data});
  }

  con.synchronize();
}

void warmup(COMB::ExecContexts& exec,
            COMB::Allocators& alloc,
            COMB::ExecutorsAvailable& exec_avail,
            Timer& tm, IdxT num_vars, IdxT len)
{
  // warm-up memory pools
  do_warmup(exec.seq, alloc.host.allocator(), tm, num_vars, len);

#ifdef COMB_ENABLE_OPENMP
  do_warmup(exec.omp, alloc.host.allocator(), tm, num_vars, len);
#endif

#ifdef COMB_ENABLE_CUDA
  do_warmup(exec.seq, alloc.cuda_hostpinned.allocator(), tm, num_vars, len);

  do_warmup(exec.cuda, alloc.cuda_device.allocator(), tm, num_vars, len);

  do_warmup(exec.seq,  alloc.cuda_managed.allocator(), tm, num_vars, len);
  do_warmup(exec.cuda, alloc.cuda_managed.allocator(), tm, num_vars, len);

  do_warmup(exec.seq,        alloc.cuda_managed_host_preferred.allocator(), tm, num_vars, len);
  if (exec_avail.cuda_batch) {
    do_warmup(exec.cuda_batch, alloc.cuda_managed_host_preferred.allocator(), tm, num_vars, len);
  } else {
    do_warmup(exec.cuda,       alloc.cuda_managed_host_preferred.allocator(), tm, num_vars, len);
  }

  do_warmup(exec.seq,             alloc.cuda_managed_host_preferred_device_accessed.allocator(), tm, num_vars, len);
  if (exec_avail.cuda_persistent) {
    do_warmup(exec.cuda_persistent, alloc.cuda_managed_host_preferred_device_accessed.allocator(), tm, num_vars, len);
  } else {
    do_warmup(exec.cuda,            alloc.cuda_managed_host_preferred_device_accessed.allocator(), tm, num_vars, len);
  }

  {
    SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

    do_warmup(exec.seq,        alloc.cuda_managed_device_preferred.allocator(), tm, num_vars, len);
    if (exec_avail.cuda_batch_fewgs) {
      do_warmup(exec.cuda_batch, alloc.cuda_managed_device_preferred.allocator(), tm, num_vars, len);
    } else {
      do_warmup(exec.cuda,       alloc.cuda_managed_device_preferred.allocator(), tm, num_vars, len);
    }

    do_warmup(exec.seq,             alloc.cuda_managed_device_preferred_host_accessed.allocator(), tm, num_vars, len);
    if (exec_avail.cuda_persistent_fewgs) {
      do_warmup(exec.cuda_persistent, alloc.cuda_managed_device_preferred_host_accessed.allocator(), tm, num_vars, len);
    } else {
      do_warmup(exec.cuda,            alloc.cuda_managed_device_preferred_host_accessed.allocator(), tm, num_vars, len);
    }
  }
#endif

#ifdef COMB_ENABLE_CUDA_GRAPH
  do_warmup(exec.cuda_graph, alloc.cuda_device.allocator(), tm, num_vars, len);
#endif
}

} // namespace COMB
