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

#ifdef COMB_ENABLE_GPUMP

#include "comm_pol_gpump.hpp"
#include "do_cycles.hpp"

namespace COMB {

void test_cycles_gpump(CommInfo& comminfo, MeshInfo& info,
                       COMB::ExecContexts& exec,
                       COMB::Allocators& alloc,
                       COMB::ExecutorsAvailable& exec_avail,
                       IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  CommContext<gpump_pol> con_comm;

  AllocatorInfo& cpu_many_aloc = alloc.cuda_device;
  AllocatorInfo& cpu_few_aloc  = alloc.cuda_device;

#ifdef COMB_ENABLE_CUDA
  AllocatorInfo& cuda_many_aloc = alloc.cuda_device;
  AllocatorInfo& cuda_few_aloc  = alloc.cuda_device;
#else
  AllocatorInfo& cuda_many_aloc = alloc.invalid;
  AllocatorInfo& cuda_few_aloc  = alloc.invalid;
#endif

  // host allocated
  if (alloc.host.available()) {
    AllocatorInfo& mesh_aloc = alloc.host;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.cuda, cuda_few_aloc.allocator(), tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.cuda_graph, cuda_few_aloc.allocator(), tm, tm_total);
#endif
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
    // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif
  }

#ifdef COMB_ENABLE_CUDA
  // host pinned allocated
  if (alloc.cuda_hostpinned.available()) {
    AllocatorInfo& mesh_aloc = alloc.cuda_hostpinned;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.cuda, cuda_few_aloc.allocator(), tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.cuda_graph, cuda_few_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
  }

  // device allocated
  if (alloc.cuda_device.available()) {
    AllocatorInfo& mesh_aloc = alloc.cuda_device;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.cuda, cuda_few_aloc.allocator(), tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.cuda_graph, cuda_few_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
  }

  // managed allocated
  if (alloc.cuda_managed.available()) {
    AllocatorInfo& mesh_aloc = alloc.cuda_managed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.cuda, cuda_few_aloc.allocator(), tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.cuda_graph, cuda_few_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
  }

  // managed host preferred allocated
  if (alloc.cuda_managed_host_preferred.available()) {
    AllocatorInfo& mesh_aloc = alloc.cuda_managed_host_preferred;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.cuda, cuda_few_aloc.allocator(), tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.cuda_graph, cuda_few_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
  }

  // managed host preferred device accessed allocated
  if (alloc.cuda_managed_host_preferred_device_accessed.available()) {
    AllocatorInfo& mesh_aloc = alloc.cuda_managed_host_preferred_device_accessed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.cuda, cuda_few_aloc.allocator(), tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.cuda_graph, cuda_few_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
  }

  // managed device preferred allocated
  if (alloc.cuda_managed_device_preferred.available()) {
    AllocatorInfo& mesh_aloc = alloc.cuda_managed_device_preferred;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.cuda, cuda_few_aloc.allocator(), tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.cuda_graph, cuda_few_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
  }

  // managed device preferred host accessed allocated
  if (alloc.cuda_managed_device_preferred_host_accessed.available()) {
    AllocatorInfo& mesh_aloc = alloc.cuda_managed_device_preferred_host_accessed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.cuda, cuda_few_aloc.allocator(), tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
        do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.cuda_graph, cuda_few_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif

    // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    //   do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
  }
#endif // COMB_ENABLE_CUDA
}

} // namespace COMB

#endif // COMB_ENABLE_GPUMP
