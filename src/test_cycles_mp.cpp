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

#ifdef COMB_ENABLE_MP

#include "comm_pol_mp.hpp"
#include "do_cycles.hpp"

namespace COMB {

void test_cycles_mp(CommInfo& comminfo, MeshInfo& info,
                       COMB::ExecContexts& exec,
                       COMB::Allocators& alloc,
                       COMB::AllocatorsAvailable& memory_avail,
                       COMB::ExecutorsAvailable& exec_avail,
                       IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  using policy_comm = mp_pol;

  // host allocated
  if (memory_avail.host) {
    COMB::Allocator& mesh_aloc = alloc.host;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (memory_avail.cuda_device_accessible_from_host) {
      if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif
    }

#ifdef COMB_ENABLE_CUDA
    if (memory_avail.cuda_host_accessible_from_device && memory_avail.cuda_device_accessible_from_host) {
      if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.seq, alloc.device, tm, tm_total);
    }

    if (memory_avail.cuda_host_accessible_from_device) {

      if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.cuda, alloc.device, tm, tm_total);

      {
        if (memory_avail.cuda_device_accessible_from_host) {
          if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
            do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);
        }

        if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
          do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

        if (memory_avail.cuda_device_accessible_from_host) {
          if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
            do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);
        }

        if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
          do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);


        SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

        if (memory_avail.cuda_device_accessible_from_host) {
          if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
            do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);
        }

        if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
          do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);
        if (memory_avail.cuda_device_accessible_from_host) {
          if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
            do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);
        }

        if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
          do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);
      }

#ifdef COMB_ENABLE_CUDA_GRAPH
      if (memory_avail.cuda_device_accessible_from_host) {
        if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
          do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.seq, alloc.device, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.cuda_graph, alloc.device, tm, tm_total);
#endif
    }
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
    //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
    //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
    if (memory_avail.cuda_host_accessible_from_device) {
      // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
    }
#endif
  }

#ifdef COMB_ENABLE_CUDA
  // host pinned allocated
  if (memory_avail.cuda_hostpinned) {
    COMB::Allocator& mesh_aloc = alloc.hostpinned;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.cuda, alloc.device, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.cuda_graph, alloc.device, tm, tm_total);
#endif

    // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
    //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
    //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
#endif

    // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
    //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
  }

  // device allocated
  if (memory_avail.cuda_device) {
    COMB::Allocator& mesh_aloc = alloc.device;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (memory_avail.cuda_device_accessible_from_host) {
      if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.seq, alloc.device, tm, tm_total);
    }

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.cuda, alloc.device, tm, tm_total);

    {
      if (memory_avail.cuda_device_accessible_from_host) {
        if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
          do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (memory_avail.cuda_device_accessible_from_host) {
        if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
          do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (memory_avail.cuda_device_accessible_from_host) {
        if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
          do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (memory_avail.cuda_device_accessible_from_host) {
        if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
          do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (memory_avail.cuda_device_accessible_from_host) {
      if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.seq, alloc.device, tm, tm_total);
    }

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.cuda_graph, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (memory_avail.cuda_device_accessible_from_host) {
        // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
      }

#ifdef COMB_ENABLE_OPENMP
      if (memory_avail.cuda_device_accessible_from_host) {
        // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
      }
#endif

      // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
    }
  }

  // managed allocated
  if (memory_avail.cuda_managed) {
    COMB::Allocator& mesh_aloc = alloc.managed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.cuda, alloc.device, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.cuda_graph, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
#endif

      // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
    }
  }

  // managed host preferred allocated
  if (memory_avail.cuda_managed_host_preferred) {
    COMB::Allocator& mesh_aloc = alloc.managed_host_preferred;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.cuda, alloc.device, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.cuda_graph, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
#endif

      // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
    }
  }

  // managed host preferred device accessed allocated
  if (memory_avail.cuda_managed_host_preferred_device_accessed) {
    COMB::Allocator& mesh_aloc = alloc.managed_host_preferred_device_accessed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.cuda, alloc.device, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.cuda_graph, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
#endif

      // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
    }
  }

  // managed device preferred allocated
  if (memory_avail.cuda_managed_device_preferred) {
    COMB::Allocator& mesh_aloc = alloc.managed_device_preferred;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.cuda, alloc.device, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.cuda_graph, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
#endif

      // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
    }
  }

  // managed device preferred host accessed allocated
  if (memory_avail.cuda_managed_device_preferred_host_accessed) {
    COMB::Allocator& mesh_aloc = alloc.managed_device_preferred_host_accessed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.seq, alloc.device, exec.seq, alloc.device, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, alloc.device, exec.omp, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, alloc.device, exec.cuda, alloc.device, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, alloc.device, exec.cuda_batch, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.seq, alloc.device, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, alloc.device, exec.cuda_persistent, alloc.device, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.seq, alloc.device, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, alloc.device, exec.cuda_graph, alloc.device, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      // if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      // if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
#endif

      // if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
      //   do_cycles<policy_comm>(comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
    }
  }
#endif // COMB_ENABLE_CUDA
}

} // namespace COMB

#endif // COMB_ENABLE_MP
