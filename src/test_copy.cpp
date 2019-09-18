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
bool should_do_copy(ExecContext<pol>& con,
                    COMB::AllocatorInfo& dst_aloc,
                    COMB::AllocatorInfo& src_aloc)
{
  return dst_aloc.available() // && src_aloc.available()
      && dst_aloc.accessible(con)
      && src_aloc.accessible(con) ;
}

template < typename pol >
void do_copy(ExecContext<pol>& con,
             CommInfo& comminfo,
             COMB::Allocator& dst_aloc,
             COMB::Allocator& src_aloc,
             Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats)
{
  tm.clear();

  char test_name[1024] = ""; snprintf(test_name, 1024, "memcpy %s dst %s src %s", pol::get_name(), dst_aloc.name(), src_aloc.name());
  fgprintf(FileGroup::all, "Starting test %s\n", test_name);

  char sub_test_name[1024] = ""; snprintf(sub_test_name, 1024, "copy_sync-%d-%d-%zu", num_vars, len, sizeof(DataT));

  Range r(test_name, Range::green);

  DataT** src = new DataT*[num_vars];
  DataT** dst = new DataT*[num_vars];

  for (IdxT i = 0; i < num_vars; ++i) {
    src[i] = (DataT*)src_aloc.allocate(len*sizeof(DataT));
    dst[i] = (DataT*)dst_aloc.allocate(len*sizeof(DataT));
  }

  // setup
  for (IdxT i = 0; i < num_vars; ++i) {
    con.for_all(0, len, detail::set_n1{dst[i]});
    con.for_all(0, len, detail::set_0{src[i]});
    con.for_all(0, len, detail::set_copy{dst[i], src[i]});
  }

  con.synchronize();

  auto g1 = con.create_group();
  auto g2 = con.create_group();
  auto c1 = con.create_component();
  auto c2 = con.create_component();

  IdxT ntestrepeats = std::max(IdxT{1}, nrepeats/IdxT{10});
  for (IdxT rep = 0; rep < ntestrepeats; ++rep) { // test comm

    con.start_group(g1);
    con.start_component(g1, c1);
    for (IdxT i = 0; i < num_vars; ++i) {
      con.for_all(0, len, detail::set_copy{src[i], dst[i]});
    }
    con.finish_component(g1, c1);
    con.finish_group(g1);

    con.synchronize();

    // tm.start(TIMER_CONTEXT, sub_test_name);

    con.start_group(g2);
    con.start_component(g2, c2);
    for (IdxT i = 0; i < num_vars; ++i) {
      con.for_all(0, len, detail::set_copy{dst[i], src[i]});
    }
    con.finish_component(g2, c2);
    con.finish_group(g2);

    con.synchronize();

    // tm.stop(TIMER_CONTEXT);
  }

  for (IdxT rep = 0; rep < nrepeats; ++rep) {

    con.start_group(g1);
    con.start_component(g1, c1);
    for (IdxT i = 0; i < num_vars; ++i) {
      con.for_all(0, len, detail::set_copy{src[i], dst[i]});
    }
    con.finish_component(g1, c1);
    con.finish_group(g1);

    con.synchronize();

    tm.start(TIMER_CONTEXT, sub_test_name);

    con.start_group(g2);
    con.start_component(g2, c2);
    for (IdxT i = 0; i < num_vars; ++i) {
      con.for_all(0, len, detail::set_copy{dst[i], src[i]});
    }
    con.finish_component(g2, c2);
    con.finish_group(g2);

    con.synchronize();

    tm.stop(TIMER_CONTEXT);
  }

  con.destroy_component(c2);
  con.destroy_component(c1);
  con.destroy_group(g2);
  con.destroy_group(g1);

  print_timer(comminfo, tm);
  tm.clear();

  for (IdxT i = 0; i < num_vars; ++i) {
    dst_aloc.deallocate(dst[i]);
    src_aloc.deallocate(src[i]);
  }

  delete[] dst;
  delete[] src;
}

void test_copy_allocator(CommInfo& comminfo,
                         COMB::ExecContexts& exec,
                         AllocatorInfo& dst_aloc,
                         AllocatorInfo& cpu_src_aloc,
                         AllocatorInfo& cuda_src_aloc,
                         COMB::ExecutorsAvailable& exec_avail,
                         Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats)
{
  char name[1024] = ""; snprintf(name, 1024, "set_vars %s", dst_aloc.allocator().name());
  Range r0(name, Range::green);

  if (exec_avail.seq && should_do_copy(exec.seq, dst_aloc, cpu_src_aloc))
    do_copy(exec.seq, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_OPENMP
  if (exec_avail.omp && should_do_copy(exec.omp, dst_aloc, cpu_src_aloc))
    do_copy(exec.omp, comminfo, dst_aloc.allocator(), cpu_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif

#ifdef COMB_ENABLE_CUDA
  if (exec_avail.cuda && should_do_copy(exec.cuda, dst_aloc, cuda_src_aloc))
    do_copy(exec.cuda, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

  if (exec_avail.cuda_batch && should_do_copy(exec.cuda_batch, dst_aloc, cuda_src_aloc))
    do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

  if (exec_avail.cuda_persistent && should_do_copy(exec.cuda_persistent, dst_aloc, cuda_src_aloc))
    do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

  {
    SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

    if (exec_avail.cuda_batch_fewgs && should_do_copy(exec.cuda_batch, dst_aloc, cuda_src_aloc))
      do_copy(exec.cuda_batch, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);

    if (exec_avail.cuda_persistent_fewgs && should_do_copy(exec.cuda_persistent, dst_aloc, cuda_src_aloc))
      do_copy(exec.cuda_persistent, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
  }

#ifdef COMB_ENABLE_CUDA_GRAPH
  if (exec_avail.cuda_graph && should_do_copy(exec.cuda_graph, dst_aloc, cuda_src_aloc))
    do_copy(exec.cuda_graph, comminfo, dst_aloc.allocator(), cuda_src_aloc.allocator(), tm, num_vars, len, nrepeats);
#endif
#else
  COMB::ignore_unused(cuda_src_aloc);
#endif
}

void test_copy_allocators(CommInfo& comminfo,
                          COMB::ExecContexts& exec,
                          COMB::Allocators& alloc,
                          AllocatorInfo& cpu_src_aloc,
                          AllocatorInfo& cuda_src_aloc,
                          COMB::ExecutorsAvailable& exec_avail,
                          Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats)
{
  test_copy_allocator(comminfo,
                      exec,
                      alloc.host,
                      cpu_src_aloc,
                      cuda_src_aloc,
                      exec_avail,
                      tm, num_vars, len, nrepeats);

#ifdef COMB_ENABLE_CUDA

  test_copy_allocator(comminfo,
                      exec,
                      alloc.cuda_hostpinned,
                      cpu_src_aloc,
                      cuda_src_aloc,
                      exec_avail,
                      tm, num_vars, len, nrepeats);

  test_copy_allocator(comminfo,
                      exec,
                      alloc.cuda_device,
                      cpu_src_aloc,
                      cuda_src_aloc,
                      exec_avail,
                      tm, num_vars, len, nrepeats);

  test_copy_allocator(comminfo,
                      exec,
                      alloc.cuda_managed,
                      cpu_src_aloc,
                      cuda_src_aloc,
                      exec_avail,
                      tm, num_vars, len, nrepeats);

  test_copy_allocator(comminfo,
                      exec,
                      alloc.cuda_managed_host_preferred,
                      cpu_src_aloc,
                      cuda_src_aloc,
                      exec_avail,
                      tm, num_vars, len, nrepeats);

  test_copy_allocator(comminfo,
                      exec,
                      alloc.cuda_managed_host_preferred_device_accessed,
                      cpu_src_aloc,
                      cuda_src_aloc,
                      exec_avail,
                      tm, num_vars, len, nrepeats);

  test_copy_allocator(comminfo,
                      exec,
                      alloc.cuda_managed_device_preferred,
                      cpu_src_aloc,
                      cuda_src_aloc,
                      exec_avail,
                      tm, num_vars, len, nrepeats);

  test_copy_allocator(comminfo,
                      exec,
                      alloc.cuda_managed_device_preferred_host_accessed,
                      cpu_src_aloc,
                      cuda_src_aloc,
                      exec_avail,
                      tm, num_vars, len, nrepeats);

#endif // COMB_ENABLE_CUDA

}

void test_copy(CommInfo& comminfo,
               COMB::ExecContexts& exec,
               COMB::Allocators& alloc,
               COMB::ExecutorsAvailable& exec_avail,
               Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats)
{

  {
    // src host memory tests
    AllocatorInfo& cpu_src_aloc = alloc.host;

#ifdef COMB_ENABLE_CUDA
    AllocatorInfo& cuda_src_aloc = alloc.cuda_hostpinned;
#else
    AllocatorInfo& cuda_src_aloc = alloc.invalid;
#endif

    test_copy_allocators(comminfo,
                         exec,
                         alloc,
                         cpu_src_aloc,
                         cuda_src_aloc,
                         exec_avail,
                         tm, num_vars, len, nrepeats);

  }

#ifdef COMB_ENABLE_CUDA
  {
    // src cuda memory tests
    AllocatorInfo& cpu_src_aloc = alloc.cuda_device;

    AllocatorInfo& cuda_src_aloc = alloc.cuda_device;

    test_copy_allocators(comminfo,
                         exec,
                         alloc,
                         cpu_src_aloc,
                         cuda_src_aloc,
                         exec_avail,
                         tm, num_vars, len, nrepeats);

  }
#endif // COMB_ENABLE_CUDA

}

} // namespace COMB
