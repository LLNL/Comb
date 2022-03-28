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

#ifndef _EXEC_HPP
#define _EXEC_HPP

#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <type_traits>

#include "exec_utils.hpp"
#include "memory.hpp"
#include "ExecContext.hpp"

#include "exec_fused.hpp"

#include "exec_pol_seq.hpp"
#include "exec_pol_omp.hpp"
#include "exec_pol_cuda.hpp"
#include "exec_pol_cuda_graph.hpp"
#include "exec_pol_hip.hpp"
#include "exec_pol_mpi_type.hpp"
#include "exec_pol_raja.hpp"

namespace COMB {

template < typename my_context_type >
struct ContextHolder
{
  using context_type = my_context_type;

  bool m_available = false;

  bool available() const
  {
    return m_available;
  }

  template < typename ... Ts >
  void create(Ts&&... args)
  {
    destroy();
    m_context = new context_type(std::forward<Ts>(args)...);
  }

  context_type& get()
  {
    assert(m_context != nullptr);
    return *m_context;
  }

  void destroy()
  {
    if (m_context) {
      delete m_context;
      m_context = nullptr;
    }
  }

  ~ContextHolder()
  {
    destroy();
  }

private:
  context_type* m_context = nullptr;
};

struct Executors
{
  Executors()
  { }

  Executors(Executors const&) = delete;
  Executors(Executors &&) = delete;
  Executors& operator=(Executors const&) = delete;
  Executors& operator=(Executors &&) = delete;

  void create_executors(Allocators& alocs)
  {
    base_cpu.create();
#ifdef COMB_ENABLE_MPI
    base_mpi.create();
#endif
#ifdef COMB_ENABLE_CUDA
    base_cuda.create();
#endif
#ifdef COMB_ENABLE_HIP
    base_hip.create();
#endif
#ifdef COMB_ENABLE_RAJA
    base_raja_cpu.create();
#ifdef COMB_ENABLE_CUDA
    base_raja_cuda.create();
#endif
#ifdef COMB_ENABLE_HIP
    base_raja_hip.create();
#endif
#endif

    seq.create(base_cpu.get(), alocs.host.allocator());
#ifdef COMB_ENABLE_OPENMP
    omp.create(base_cpu.get(), alocs.host.allocator());
#endif
#ifdef COMB_ENABLE_CUDA
    cuda.create(base_cuda.get(), (alocs.access.use_device_preferred_for_cuda_util_aloc) ? alocs.cuda_managed_device_preferred_host_accessed.allocator() : alocs.cuda_hostpinned.allocator());
#endif
#ifdef COMB_ENABLE_CUDA_GRAPH
    cuda_graph.create(base_cuda.get(), (alocs.access.use_device_preferred_for_cuda_util_aloc) ? alocs.cuda_managed_device_preferred_host_accessed.allocator() : alocs.cuda_hostpinned.allocator());
#endif
#ifdef COMB_ENABLE_HIP
    hip.create(base_hip.get(), (alocs.access.use_device_for_hip_util_aloc) ? alocs.hip_device.allocator() : alocs.hip_hostpinned.allocator());
#endif
#ifdef COMB_ENABLE_MPI
    mpi_type.create(base_mpi.get(), alocs.host.allocator());
#endif
#ifdef COMB_ENABLE_RAJA
    raja_seq.create(base_raja_cpu.get(), alocs.host.allocator());
#ifdef COMB_ENABLE_OPENMP
    raja_omp.create(base_raja_cpu.get(), alocs.host.allocator());
#endif
#ifdef COMB_ENABLE_CUDA
    raja_cuda.create(base_raja_cuda.get(), (alocs.access.use_device_preferred_for_cuda_util_aloc) ? alocs.cuda_managed_device_preferred_host_accessed.allocator() : alocs.cuda_hostpinned.allocator());
#endif
#ifdef COMB_ENABLE_HIP
    raja_hip.create(base_raja_hip.get(), (alocs.access.use_device_for_hip_util_aloc) ? alocs.hip_device.allocator() : alocs.hip_hostpinned.allocator());
#endif
#endif
  }

  ContextHolder<CPUContext> base_cpu;
#ifdef COMB_ENABLE_MPI
  ContextHolder<MPIContext> base_mpi;
#endif
#ifdef COMB_ENABLE_CUDA
  ContextHolder<CudaContext> base_cuda;
#endif
#ifdef COMB_ENABLE_HIP
  ContextHolder<HipContext> base_hip;
#endif
#ifdef COMB_ENABLE_RAJA
  ContextHolder<RAJAContext<RAJA::resources::Host>> base_raja_cpu;
#ifdef COMB_ENABLE_CUDA
  ContextHolder<RAJAContext<RAJA::resources::Cuda>> base_raja_cuda;
#endif
#ifdef COMB_ENABLE_HIP
  ContextHolder<RAJAContext<RAJA::resources::Hip>> base_raja_hip;
#endif
#endif

  ContextHolder<ExecContext<seq_pol>> seq;
#ifdef COMB_ENABLE_OPENMP
  ContextHolder<ExecContext<omp_pol>> omp;
#endif
#ifdef COMB_ENABLE_CUDA
  ContextHolder<ExecContext<cuda_pol>> cuda;
#ifdef COMB_ENABLE_CUDA_GRAPH
  ContextHolder<ExecContext<cuda_graph_pol>> cuda_graph;
#endif
#endif
#ifdef COMB_ENABLE_HIP
  ContextHolder<ExecContext<hip_pol>> hip;
#endif
#ifdef COMB_ENABLE_MPI
  ContextHolder<ExecContext<mpi_type_pol>> mpi_type;
#endif
#ifdef COMB_ENABLE_RAJA
  ContextHolder<ExecContext<raja_seq_pol>> raja_seq;
#ifdef COMB_ENABLE_OPENMP
  ContextHolder<ExecContext<raja_omp_pol>> raja_omp;
#endif
#ifdef COMB_ENABLE_CUDA
  ContextHolder<ExecContext<raja_cuda_pol>> raja_cuda;
#endif
#ifdef COMB_ENABLE_HIP
  ContextHolder<ExecContext<raja_hip_pol>> raja_hip;
#endif
#endif
};

} // namespace COMB

#endif // _EXEC_HPP
