//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
void do_copy(CommInfo& comminfo, COMB::Allocator& src_aloc, COMB::Allocator& dst_aloc, Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats)
{
  tm.clear();

  char test_name[1024] = ""; snprintf(test_name, 1024, "memcpy %s dst %s src %s", pol::get_name(), dst_aloc.name(), src_aloc.name());
  comminfo.print(FileGroup::all, "Starting test %s\n", test_name);

  Range r(test_name, Range::green);

  DataT** src = new DataT*[num_vars];
  DataT** dst = new DataT*[num_vars];

  for (IdxT i = 0; i < num_vars; ++i) {
    src[i] = (DataT*)src_aloc.allocate(len*sizeof(DataT));
    dst[i] = (DataT*)dst_aloc.allocate(len*sizeof(DataT));
  }

  // setup
  for (IdxT i = 0; i < num_vars; ++i) {
    for_all(ExecContext<pol>{}, 0, len, detail::set_n1{dst[i]});
    for_all(ExecContext<pol>{}, 0, len, detail::set_0{src[i]});
    for_all(ExecContext<pol>{}, 0, len, detail::set_copy{dst[i], src[i]});
  }

  synchronize(ExecContext<pol>{});

  char sub_test_name[1024] = ""; snprintf(sub_test_name, 1024, "copy_sync-%d-%d-%zu", num_vars, len, sizeof(DataT));

  for (IdxT rep = 0; rep < nrepeats; ++rep) {

    for (IdxT i = 0; i < num_vars; ++i) {
      for_all(ExecContext<pol>{}, 0, len, detail::set_copy{src[i], dst[i]});
    }

    synchronize(ExecContext<pol>{});

    tm.start(sub_test_name);

    for (IdxT i = 0; i < num_vars; ++i) {
      for_all(ExecContext<pol>{}, 0, len, detail::set_copy{dst[i], src[i]});
    }

    synchronize(ExecContext<pol>{});

    tm.stop();
  }

  print_timer(comminfo, tm);
  tm.clear();

  for (IdxT i = 0; i < num_vars; ++i) {
    dst_aloc.deallocate(dst[i]);
    src_aloc.deallocate(src[i]);
  }

  delete[] dst;
  delete[] src;
}

void test_copy(CommInfo& comminfo,
               COMB::Allocators& alloc,
               COMB::AllocatorsAvailable& memory_avail,
               COMB::ExecutorsAvailable& exec_avail,
               Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats)
{
  // host memory
  if (memory_avail.host) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.host.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, len, nrepeats);
#endif

#ifdef COMB_ENABLE_CUDA
    if (memory_avail.cuda_host_accessible_from_device) {

      if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, len, nrepeats);

      {
        SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

        if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, len, nrepeats);

        if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, len, nrepeats);
      }

#ifdef COMB_ENABLE_CUDA_GRAPH
      if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, len, nrepeats);
#endif
    }
#endif
  }

#ifdef COMB_ENABLE_CUDA
  if (memory_avail.cuda_hostpinned) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.hostpinned.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.hostpinned, alloc.host, tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.hostpinned, alloc.host, tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, len, nrepeats);
#endif
  }

  if (memory_avail.cuda_device) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.device.name());
    Range r0(name, Range::green);

    if (memory_avail.cuda_device_accessible_from_host) {
      if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.device, alloc.host, tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.device, alloc.host, tm, num_vars, len, nrepeats);
#endif
    }

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, len, nrepeats);
#endif
  }

  if (memory_avail.cuda_managed) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed, alloc.host, tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed, alloc.host, tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, len, nrepeats);
#endif
  }

  if (memory_avail.cuda_managed_host_preferred) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed_host_preferred.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed_host_preferred, alloc.host, tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed_host_preferred, alloc.host, tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);
#endif
  }

  if (memory_avail.cuda_managed_host_preferred_device_accessed) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed_host_preferred_device_accessed.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.host, tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.host, tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);
#endif
  }

  if (memory_avail.cuda_managed_device_preferred) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed_device_preferred.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed_device_preferred, alloc.host, tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed_device_preferred, alloc.host, tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, len, nrepeats);
#endif
  }

  if (memory_avail.cuda_managed_device_preferred_host_accessed) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed_device_preferred_host_accessed.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.host, tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.host, tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, len, nrepeats);
#endif
  }
#endif // COMB_ENABLE_CUDA
}

} // namespace COMB
