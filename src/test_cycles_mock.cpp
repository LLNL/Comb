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

#include "do_cycles.hpp"

namespace COMB {

void test_cycles_mock(CommInfo& comminfo, MeshInfo& info,
                     COMB::Allocators& alloc,
                     COMB::AllocatorsAvailable& memory_avail,
                     COMB::ExecutorsAvailable& exec_avail,
                     IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  using policy_comm = mock_pol;

  // host allocated
  if (memory_avail.host) {
    COMB::Allocator& mesh_aloc = alloc.host;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
    if (memory_avail.cuda_host_accessible_from_device) {
      if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
        do_cycles<cuda_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
        do_cycles<cuda_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
        do_cycles<cuda_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
        do_cycles<cuda_pol, cuda_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
        do_cycles<cuda_pol, cuda_pol, cuda_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      {
        if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
          do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
          do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
          do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
          do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);


        SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

        if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
          do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
          do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
          do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
          do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
      }

#ifdef COMB_ENABLE_CUDA_GRAPH
      if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
        do_cycles<cuda_pol, cuda_graph_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
        do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif
    }
#endif

    if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<seq_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<omp_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
    if (memory_avail.cuda_host_accessible_from_device) {
      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
#endif
  }

#ifdef COMB_ENABLE_CUDA
  // host pinned allocated
  if (memory_avail.cuda_pinned) {
    COMB::Allocator& mesh_aloc = alloc.hostpinned;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<seq_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<omp_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
  }

  // device allocated
  if (memory_avail.cuda_device) {
    COMB::Allocator& mesh_aloc = alloc.device;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (detail::cuda::get_device_accessible_from_host()) {
      if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
        do_cycles<seq_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
        do_cycles<omp_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
        do_cycles<omp_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
        do_cycles<omp_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
        do_cycles<cuda_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
        do_cycles<cuda_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
        do_cycles<cuda_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
        do_cycles<cuda_pol, cuda_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
    }

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
          do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
          do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
          do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
          do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (detail::cuda::get_device_accessible_from_host()) {
      if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
        do_cycles<cuda_pol, cuda_graph_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
    }

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
          do_cycles<seq_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
      }

#ifdef COMB_ENABLE_OPENMP
      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
          do_cycles<omp_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
      }
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed allocated
  if (memory_avail.cuda_managed) {
    COMB::Allocator& mesh_aloc = alloc.managed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed host preferred allocated
  if (memory_avail.cuda_managed_host_preferred) {
    COMB::Allocator& mesh_aloc = alloc.managed_host_preferred;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed host preferred device accessed allocated
  if (memory_avail.cuda_managed_host_preferred_device_accessed) {
    COMB::Allocator& mesh_aloc = alloc.managed_host_preferred_device_accessed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed device preferred allocated
  if (memory_avail.cuda_managed_device_preferred) {
    COMB::Allocator& mesh_aloc = alloc.managed_device_preferred;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed device preferred host accessed allocated
  if (memory_avail.cuda_managed_device_preferred_host_accessed) {
    COMB::Allocator& mesh_aloc = alloc.managed_device_preferred_host_accessed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol, policy_comm>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }
#endif // COMB_ENABLE_CUDA
}

} // namespace COMB
