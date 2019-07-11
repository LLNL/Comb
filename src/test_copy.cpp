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
void do_copy(ExecContext<pol> const& con,
             CommInfo& comminfo,
             COMB::Allocator& src_aloc,
             COMB::Allocator& dst_aloc,
             Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats)
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
    for_all(con, 0, len, detail::set_n1{dst[i]});
    for_all(con, 0, len, detail::set_0{src[i]});
    for_all(con, 0, len, detail::set_copy{dst[i], src[i]});
  }

  synchronize(con);

  char sub_test_name[1024] = ""; snprintf(sub_test_name, 1024, "copy_sync-%d-%d-%zu", num_vars, len, sizeof(DataT));

  for (IdxT rep = 0; rep < nrepeats; ++rep) {

    for (IdxT i = 0; i < num_vars; ++i) {
      for_all(con, 0, len, detail::set_copy{src[i], dst[i]});
    }

    synchronize(con);

    tm.start(sub_test_name);

    for (IdxT i = 0; i < num_vars; ++i) {
      for_all(con, 0, len, detail::set_copy{dst[i], src[i]});
    }

    synchronize(con);

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
               COMB::ExecContexts& exec,
               COMB::Allocators& alloc,
               COMB::ExecutorsAvailable& exec_avail,
               Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats)
{
  AllocatorInfo& cpu_src_aloc = alloc.host;

#ifdef COMB_ENABLE_CUDA
  AllocatorInfo& cuda_src_aloc = alloc.cuda_hostpinned;
#endif

  // host memory
  if (alloc.host.available()) {
    AllocatorInfo& dst_aloc = alloc.host;

    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.host.allocator().name());
    Range r0(name, Range::green);

    if (exec_avail.seq)
      do_copy(exec.seq, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp)
      do_copy(exec.omp, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif

#ifdef COMB_ENABLE_CUDA
    if (alloc.access.cuda_host_accessible_from_device) {

      if (exec_avail.cuda)
        do_copy(exec.cuda, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_batch)
        do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent)
        do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      {
        SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

        if (exec_avail.cuda_batch_fewgs)
          do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

        if (exec_avail.cuda_persistent_fewgs)
          do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
      }

#ifdef COMB_ENABLE_CUDA_GRAPH
      if (exec_avail.cuda_graph)
        do_copy(exec.cuda_graph, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
    }
#endif
  }

#ifdef COMB_ENABLE_CUDA
  if (alloc.cuda_hostpinned.available()) {
    AllocatorInfo& dst_aloc = alloc.cuda_hostpinned;

    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.cuda_hostpinned.allocator().name());
    Range r0(name, Range::green);

    if (exec_avail.seq)
      do_copy(exec.seq, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp)
      do_copy(exec.omp, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda)
      do_copy(exec.cuda, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch)
      do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent)
      do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs)
        do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs)
        do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph)
      do_copy(exec.cuda_graph, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
  }

  if (alloc.cuda_device.available()) {
    AllocatorInfo& dst_aloc = alloc.cuda_device;

    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.cuda_device.allocator().name());
    Range r0(name, Range::green);

    if (alloc.access.cuda_device_accessible_from_host) {
      if (exec_avail.seq)
        do_copy(exec.seq, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp)
        do_copy(exec.omp, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
    }

    if (exec_avail.cuda)
      do_copy(exec.cuda, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch)
      do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent)
      do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs)
        do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs)
        do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph)
      do_copy(exec.cuda_graph, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
  }

  if (alloc.cuda_managed.available()) {
    AllocatorInfo& dst_aloc = alloc.cuda_managed;

    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.cuda_managed.allocator().name());
    Range r0(name, Range::green);

    if (exec_avail.seq)
      do_copy(exec.seq, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp)
      do_copy(exec.omp, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda)
      do_copy(exec.cuda, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch)
      do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent)
      do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs)
        do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs)
        do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph)
      do_copy(exec.cuda_graph, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
  }

  if (alloc.cuda_managed_host_preferred.available()) {
    AllocatorInfo& dst_aloc = alloc.cuda_managed_host_preferred;

    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.cuda_managed_host_preferred.allocator().name());
    Range r0(name, Range::green);

    if (exec_avail.seq)
      do_copy(exec.seq, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp)
      do_copy(exec.omp, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda)
      do_copy(exec.cuda, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch)
      do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent)
      do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs)
        do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs)
        do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph)
      do_copy(exec.cuda_graph, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
  }

  if (alloc.cuda_managed_host_preferred_device_accessed.available()) {
    AllocatorInfo& dst_aloc = alloc.cuda_managed_host_preferred_device_accessed;

    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.cuda_managed_host_preferred_device_accessed.allocator().name());
    Range r0(name, Range::green);

    if (exec_avail.seq)
      do_copy(exec.seq, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp)
      do_copy(exec.omp, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda)
      do_copy(exec.cuda, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch)
      do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent)
      do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs)
        do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs)
        do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph)
      do_copy(exec.cuda_graph, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
  }

  if (alloc.cuda_managed_device_preferred.available()) {
    AllocatorInfo& dst_aloc = alloc.cuda_managed_device_preferred;

    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.cuda_managed_device_preferred.allocator().name());
    Range r0(name, Range::green);

    if (exec_avail.seq)
      do_copy(exec.seq, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp)
      do_copy(exec.omp, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda)
      do_copy(exec.cuda, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch)
      do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent)
      do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs)
        do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs)
        do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph)
      do_copy(exec.cuda_graph, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
  }

  if (alloc.cuda_managed_device_preferred_host_accessed.available()) {
    AllocatorInfo& dst_aloc = alloc.cuda_managed_device_preferred_host_accessed;

    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.cuda_managed_device_preferred_host_accessed.allocator().name());
    Range r0(name, Range::green);

    if (exec_avail.seq)
      do_copy(exec.seq, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp)
      do_copy(exec.omp, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif

    if (exec_avail.cuda)
      do_copy(exec.cuda, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_batch)
      do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent)
      do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs)
        do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

      if (exec_avail.cuda_persistent_fewgs)
        do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph)
      do_copy(exec.cuda_graph, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
  }
#endif // COMB_ENABLE_CUDA
}

} // namespace COMB
