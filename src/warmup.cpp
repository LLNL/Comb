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

namespace COMB {

template < typename pol >
void do_warmup(ExecContext<pol>& con, COMB::Allocator& aloc, Timer& tm, IdxT num_vars, IdxT len)
{
  tm.clear();

  char test_name[1024] = ""; snprintf(test_name, 1024, "warmup %s %s", pol::get_name(), aloc.name());
  Range r(test_name, Range::green);

  std::vector<DataT*> vars(num_vars, nullptr);

  for (IdxT i = 0; i < num_vars; ++i) {
    vars[i] = (DataT*)aloc.allocate(len*sizeof(DataT));
  }

  for (IdxT i = 0; i < num_vars; ++i) {

    DataT* data = vars[i];

    con.for_all(len, detail::set_n1{data});
  }

  con.synchronize();

  for (IdxT i = 0; i < num_vars; ++i) {
    aloc.deallocate(vars[i]);
  }
}

void warmup(COMB::Executors& exec,
            COMB::Allocators& alloc,
            Timer& tm, IdxT num_vars, IdxT len)
{
  // warm-up memory pools
  do_warmup(exec.seq.get(), alloc.host.allocator(), tm, num_vars, len);

#ifdef COMB_ENABLE_OPENMP
  do_warmup(exec.omp.get(), alloc.host.allocator(), tm, num_vars, len);
#endif

#ifdef COMB_ENABLE_CUDA
  do_warmup(exec.seq.get(), alloc.cuda_hostpinned.allocator(), tm, num_vars, len);

  do_warmup(exec.cuda.get(), alloc.cuda_device.allocator(), tm, num_vars, len);

  do_warmup(exec.seq.get(),  alloc.cuda_managed.allocator(), tm, num_vars, len);
  do_warmup(exec.cuda.get(), alloc.cuda_managed.allocator(), tm, num_vars, len);

  if (alloc.cuda_managed_host_preferred.available()) {
    do_warmup(exec.seq.get(),  alloc.cuda_managed_host_preferred.allocator(),   tm, num_vars, len);
    do_warmup(exec.cuda.get(), alloc.cuda_managed_device_preferred.allocator(), tm, num_vars, len);
  }

  if (alloc.cuda_managed_host_preferred_device_accessed.available()) {
    do_warmup(exec.seq.get(),  alloc.cuda_managed_host_preferred_device_accessed.allocator(), tm, num_vars, len);
    do_warmup(exec.cuda.get(), alloc.cuda_managed_device_preferred_host_accessed.allocator(), tm, num_vars, len);
  }
#endif

#ifdef COMB_ENABLE_CUDA_GRAPH
  do_warmup(exec.cuda_graph.get(), alloc.cuda_device.allocator(), tm, num_vars, len);
#endif

#ifdef COMB_ENABLE_RAJA
  do_warmup(exec.raja_seq.get(), alloc.host.allocator(), tm, num_vars, len);

#ifdef COMB_ENABLE_OPENMP
  do_warmup(exec.raja_omp.get(), alloc.host.allocator(), tm, num_vars, len);
#endif

#ifdef COMB_ENABLE_CUDA
  do_warmup(exec.raja_cuda.get(), alloc.cuda_device.allocator(), tm, num_vars, len);
#endif
#endif
}

} // namespace COMB
