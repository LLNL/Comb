//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
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

#ifndef _DO_CYCLES_HPP
#define _DO_CYCLES_HPP

#include <type_traits>

#include "comb.hpp"
#include "CommFactory.hpp"

namespace COMB {

// Note that do_cycles instantiations are generated manually in cmake by
// configuring do_cycles.cpp.in with each of the policy combinations used.
// This speeds up compile times by increasing the number of independent files.
template < typename pol_comm, typename exec_mesh, typename exec_many, typename exec_few >
extern void do_cycles(
    CommContext<pol_comm>& con_comm_in,
    CommInfo& comm_info, MeshInfo& info,
    IdxT num_vars, IdxT ncycles,
    ContextHolder<exec_mesh>& con_mesh_in, AllocatorInfo& aloc_mesh_in,
    ContextHolder<exec_many>& con_many_in, AllocatorInfo& aloc_many_in,
    ContextHolder<exec_few>& con_few_in,   AllocatorInfo& aloc_few_in,
    Timer& tm, Timer& tm_total);


#ifdef COMB_ENABLE_MPI

template < typename comm_pol >
void do_cycles_mpi_type(std::true_type const&,
                        CommContext<comm_pol>& con_comm,
                        CommInfo& comminfo, MeshInfo& info,
                        COMB::Executors& exec,
                        AllocatorInfo& mesh_aloc,
                        IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc, tm, tm_total);
#endif
}

template < typename comm_pol >
void do_cycles_mpi_type(std::false_type const&,
                        CommContext<comm_pol>&,
                        CommInfo&, MeshInfo&,
                        COMB::Executors&,
                        AllocatorInfo&,
                        IdxT, IdxT, Timer&, Timer&)
{
}

#endif

template < typename comm_pol >
void do_cycles_allocator(CommContext<comm_pol>& con_comm,
                         CommInfo& comminfo, MeshInfo& info,
                         COMB::Executors& exec,
                         AllocatorInfo& mesh_aloc,
                         AllocatorInfo& cpu_many_aloc, AllocatorInfo& cpu_few_aloc,
                         AllocatorInfo& cuda_many_aloc, AllocatorInfo& cuda_few_aloc,
                         IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
  Range r0(name, Range::blue);

  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc, tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc, tm, tm_total);
#endif

  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc, tm, tm_total);

  {
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc, tm, tm_total);

    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc, tm, tm_total);


    SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch_fewgs, cuda_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_batch_fewgs, cuda_many_aloc, exec.cuda_batch_fewgs, cuda_few_aloc, tm, tm_total);

    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent_fewgs, cuda_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_persistent_fewgs, cuda_many_aloc, exec.cuda_persistent_fewgs, cuda_few_aloc, tm, tm_total);
  }

#ifdef COMB_ENABLE_CUDA_GRAPH
  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc, tm, tm_total);

  do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc, tm, tm_total);
#endif
#else
  COMB::ignore_unused(cuda_many_aloc, cuda_few_aloc);
#endif

#ifdef COMB_ENABLE_MPI
  do_cycles_mpi_type(typename std::conditional<comm_pol::use_mpi_type, std::true_type, std::false_type>::type{},
      con_comm, comminfo, info, exec, mesh_aloc, num_vars, ncycles, tm, tm_total);
#endif
}


template < typename comm_pol >
void do_cycles_allocators(CommContext<comm_pol>& con_comm,
                          CommInfo& comminfo, MeshInfo& info,
                          COMB::Executors& exec,
                          Allocators& alloc,
                          AllocatorInfo& cpu_many_aloc, AllocatorInfo& cpu_few_aloc,
                          AllocatorInfo& cuda_many_aloc, AllocatorInfo& cuda_few_aloc,
                          IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.host,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      num_vars, ncycles, tm, tm_total);

#ifdef COMB_ENABLE_CUDA

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_hostpinned,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_device,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed_host_preferred,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed_host_preferred_device_accessed,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed_device_preferred,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed_device_preferred_host_accessed,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      num_vars, ncycles, tm, tm_total);

#endif // COMB_ENABLE_CUDA

}

} // namespace COMB

#endif // _DO_CYCLES_HPP

