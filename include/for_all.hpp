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

#ifndef _FOR_ALL_HPP
#define _FOR_ALL_HPP

#include "config.hpp"

#include <cstdio>
#include <cstdlib>

#include <type_traits>

#include "utils.hpp"
#include "memory.hpp"
#include "ExecContext.hpp"


inline bool& comb_allow_per_message_pack_fusing()
{
  static bool allow = true;
  return allow;
}

inline bool& comb_allow_pack_loop_fusion()
{
  static bool allow = true;
  return allow;
}

namespace detail {

template < typename body_type >
struct adapter_2d {
  IdxT begin0, begin1;
  IdxT len1;
  body_type body;
  template < typename body_type_ >
  adapter_2d(IdxT begin0_, IdxT end0_, IdxT begin1_, IdxT end1_, body_type_&& body_)
    : begin0(begin0_)
    , begin1(begin1_)
    , len1(end1_ - begin1_)
    , body(std::forward<body_type_>(body_))
  { COMB::ignore_unused(end0_); }
  COMB_HOST COMB_DEVICE
  void operator() (IdxT, IdxT idx) const
  {
    IdxT i0 = idx / len1;
    IdxT i1 = idx - i0 * len1;

    //FGPRINTF(FileGroup::proc, "adapter_2d (%i+%i %i+%i)%i\n", i0, begin0, i1, begin1, idx);
    //assert(0 <= i0 + begin0 && i0 + begin0 < 3);
    //assert(0 <= i1 + begin1 && i1 + begin1 < 3);

    body(i0 + begin0, i1 + begin1, idx);
  }
};

template < typename body_type >
struct adapter_3d {
  IdxT begin0, begin1, begin2;
  IdxT len2, len12;
  body_type body;
  template < typename body_type_ >
  adapter_3d(IdxT begin0_, IdxT end0_, IdxT begin1_, IdxT end1_, IdxT begin2_, IdxT end2_, body_type_&& body_)
    : begin0(begin0_)
    , begin1(begin1_)
    , begin2(begin2_)
    , len2(end2_ - begin2_)
    , len12((end1_ - begin1_) * (end2_ - begin2_))
    , body(std::forward<body_type_>(body_))
  { COMB::ignore_unused(end0_); }
  COMB_HOST COMB_DEVICE
  void operator() (IdxT, IdxT idx) const
  {
    IdxT i0 = idx / len12;
    IdxT idx12 = idx - i0 * len12;

    IdxT i1 = idx12 / len2;
    IdxT i2 = idx12 - i1 * len2;

    //FGPRINTF(FileGroup::proc, "adapter_3d (%i+%i %i+%i %i+%i)%i\n", i0, begin0, i1, begin1, i2, begin2, idx);
    //assert(0 <= i0 + begin0 && i0 + begin0 < 3);
    //assert(0 <= i1 + begin1 && i1 + begin1 < 3);
    //assert(0 <= i2 + begin2 && i2 + begin2 < 13);

    body(i0 + begin0, i1 + begin1, i2 + begin2, idx);
  }
};

} // namespace detail

#include "pol_seq.hpp"
#include "pol_omp.hpp"
#include "pol_cuda.hpp"
#include "pol_cuda_batch.hpp"
#include "pol_cuda_persistent.hpp"
#include "pol_cuda_graph.hpp"
#include "pol_mpi_type.hpp"

namespace COMB {

struct ExecutorsAvailable
{
  bool seq = false;
  bool omp = false;
  bool cuda = false;
  bool cuda_batch = false;
  bool cuda_batch_fewgs = false;
  bool cuda_persistent = false;
  bool cuda_persistent_fewgs = false;
  bool cuda_graph = false;
  bool mpi_type = false;
};

struct ExecContexts
{
  CPUContext base_cpu{};
#ifdef COMB_ENABLE_MPI
  MPIContext base_mpi{};
#endif
#ifdef COMB_ENABLE_CUDA
  CudaContext base_cuda{};
#endif

  ExecContext<seq_pol> seq;
#ifdef COMB_ENABLE_OPENMP
  ExecContext<omp_pol> omp;
#endif
#ifdef COMB_ENABLE_CUDA
  ExecContext<cuda_pol> cuda;
  ExecContext<cuda_batch_pol> cuda_batch;
  ExecContext<cuda_persistent_pol> cuda_persistent;
#endif
#ifdef COMB_ENABLE_CUDA_GRAPH
  ExecContext<cuda_graph_pol> cuda_graph;
#endif
#ifdef COMB_ENABLE_MPI
  ExecContext<mpi_type_pol> mpi_type;
#endif

  ExecContexts(Allocators& alocs)
    : seq(base_cpu, alocs.host.allocator())
#ifdef COMB_ENABLE_OPENMP
    , omp(base_cpu, alocs.host.allocator())
#endif
#ifdef COMB_ENABLE_CUDA
    , cuda(base_cuda, (alocs.access.use_device_preferred_for_cuda_util_aloc) ? alocs.cuda_managed_device_preferred_host_accessed.allocator() : alocs.cuda_hostpinned.allocator())
    , cuda_batch(base_cuda, (alocs.access.use_device_preferred_for_cuda_util_aloc) ? alocs.cuda_managed_device_preferred_host_accessed.allocator() : alocs.cuda_hostpinned.allocator())
    , cuda_persistent(base_cuda, (alocs.access.use_device_preferred_for_cuda_util_aloc) ? alocs.cuda_managed_device_preferred_host_accessed.allocator() : alocs.cuda_hostpinned.allocator())
#endif
#ifdef COMB_ENABLE_CUDA_GRAPH
    , cuda_graph(base_cuda, (alocs.access.use_device_preferred_for_cuda_util_aloc) ? alocs.cuda_managed_device_preferred_host_accessed.allocator() : alocs.cuda_hostpinned.allocator())
#endif
#ifdef COMB_ENABLE_MPI
    , mpi_type(base_mpi, alocs.host.allocator())
#endif
  {

  }
};

} // namespace COMB

#endif // _FOR_ALL_HPP
