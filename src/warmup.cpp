//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2022, Lawrence Livermore National Security, LLC.
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
  // find an available cuda allocator for use later
  COMB::AllocatorInfo* cuda_alloc = nullptr;

  if (alloc.cuda_device.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.cuda_device.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.cuda.get(), alloc.cuda_device.allocator(), tm, num_vars, len);
    if (!cuda_alloc) { cuda_alloc = &alloc.cuda_device; }
  }

  if (alloc.cuda_hostpinned.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.cuda_hostpinned.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.seq.get(), alloc.cuda_hostpinned.allocator(), tm, num_vars, len);
    if (!cuda_alloc) { cuda_alloc = &alloc.cuda_hostpinned; }
  }

  if (alloc.cuda_managed.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.cuda_managed.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.seq.get(),  alloc.cuda_managed.allocator(), tm, num_vars, len);
    do_warmup(exec.cuda.get(), alloc.cuda_managed.allocator(), tm, num_vars, len);
    if (!cuda_alloc) { cuda_alloc = &alloc.cuda_managed; }
  }

  if (alloc.cuda_managed_host_preferred.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.cuda_managed_host_preferred.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.seq.get(),  alloc.cuda_managed_host_preferred.allocator(),   tm, num_vars, len);
    do_warmup(exec.cuda.get(), alloc.cuda_managed_host_preferred.allocator(), tm, num_vars, len);
    if (!cuda_alloc) { cuda_alloc = &alloc.cuda_managed_host_preferred; }
  }

  if (alloc.cuda_managed_device_preferred.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.cuda_managed_device_preferred.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.seq.get(),  alloc.cuda_managed_device_preferred.allocator(),   tm, num_vars, len);
    do_warmup(exec.cuda.get(), alloc.cuda_managed_device_preferred.allocator(), tm, num_vars, len);
    if (!cuda_alloc) { cuda_alloc = &alloc.cuda_managed_device_preferred; }
  }

  if (alloc.cuda_managed_host_preferred_device_accessed.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.cuda_managed_host_preferred_device_accessed.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.seq.get(),  alloc.cuda_managed_host_preferred_device_accessed.allocator(), tm, num_vars, len);
    do_warmup(exec.cuda.get(), alloc.cuda_managed_host_preferred_device_accessed.allocator(), tm, num_vars, len);
    if (!cuda_alloc) { cuda_alloc = &alloc.cuda_managed_host_preferred_device_accessed; }
  }

  if (alloc.cuda_managed_device_preferred_host_accessed.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.cuda_managed_device_preferred_host_accessed.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.seq.get(),  alloc.cuda_managed_device_preferred_host_accessed.allocator(), tm, num_vars, len);
    do_warmup(exec.cuda.get(), alloc.cuda_managed_device_preferred_host_accessed.allocator(), tm, num_vars, len);
    if (!cuda_alloc) { cuda_alloc = &alloc.cuda_managed_device_preferred_host_accessed; }
  }

  if (alloc.host.accessible(exec.seq.get())) {
    if (!cuda_alloc) { cuda_alloc = &alloc.host; }
  }
#endif

#ifdef COMB_ENABLE_CUDA_GRAPH
  if (cuda_alloc) {
    do_warmup(exec.cuda_graph.get(), cuda_alloc->allocator(), tm, num_vars, len);
  }
#endif

#ifdef COMB_ENABLE_HIP
  // find an available hip allocator for use later
  COMB::AllocatorInfo* hip_alloc = nullptr;

  if (alloc.hip_device.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.hip_device.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.hip.get(), alloc.hip_device.allocator(), tm, num_vars, len);
    if (!hip_alloc) { hip_alloc = &alloc.hip_device; }
  }

  if (alloc.hip_hostpinned.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.hip_hostpinned.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.seq.get(), alloc.hip_hostpinned.allocator(), tm, num_vars, len);
    if (!hip_alloc) { hip_alloc = &alloc.hip_hostpinned; }
  }

  if (alloc.hip_managed.available(COMB::AllocatorInfo::UseType::Mesh)
   || alloc.hip_managed.available(COMB::AllocatorInfo::UseType::Buffer)) {
    do_warmup(exec.seq.get(),  alloc.hip_managed.allocator(), tm, num_vars, len);
    do_warmup(exec.hip.get(), alloc.hip_managed.allocator(), tm, num_vars, len);
    if (!hip_alloc) { hip_alloc = &alloc.hip_managed; }
  }

  if (alloc.host.accessible(exec.seq.get())) {
    if (!hip_alloc) { hip_alloc = &alloc.host; }
  }
#endif

#ifdef COMB_ENABLE_RAJA
  do_warmup(exec.raja_seq.get(), alloc.host.allocator(), tm, num_vars, len);

#ifdef COMB_ENABLE_OPENMP
  do_warmup(exec.raja_omp.get(), alloc.host.allocator(), tm, num_vars, len);
#endif

#ifdef COMB_ENABLE_CUDA
  if (cuda_alloc) {
    do_warmup(exec.raja_cuda.get(), cuda_alloc->allocator(), tm, num_vars, len);
  }
#endif

#ifdef COMB_ENABLE_HIP
  if (hip_alloc) {
    do_warmup(exec.raja_hip.get(), hip_alloc->allocator(), tm, num_vars, len);
  }
#endif
#endif
}

} // namespace COMB
