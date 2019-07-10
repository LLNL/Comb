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

#ifndef _EXECCONTEXT_HPP
#define _EXECCONTEXT_HPP

#include "config.hpp"

#include "utils_cuda.hpp"

enum struct ContextEnum
{
  invalid = 0
 ,cpu = 1
 ,cuda = 2
};

struct CPUContext;
struct MPIContext;

#ifdef COMB_ENABLE_CUDA

namespace detail {

struct CudaStream
{
  CudaStream()
    : m_ref(0)
  {
    cudaCheck(cudaStreamCreateWithFlags(&m_stream, cudaStreamDefault));
    cudaCheck(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
  }

  ~CudaStream()
  {
    cudaCheck(cudaEventDestroy(m_event));
    cudaCheck(cudaStreamDestroy(m_stream));
  }

  void waitOn(CudaStream& other)
  {
    cudaCheck(cudaEventRecord(other.m_event, other.m_stream));
    cudaCheck(cudaStreamWaitEvent(m_stream, other.m_event, 0));
  }

  void synchronize()
  {
    cudaCheck(cudaStreamSynchronize(m_stream));
  }

  cudaStream_t stream()
  {
    return m_stream;
  }

  int inc()
  {
    return ++m_ref;
  }

  int dec()
  {
    return --m_ref;
  }
private:
  cudaStream_t m_stream;
  cudaEvent_t m_event;
  size_t m_ref;
};

} // namesapce detail

struct CudaContext
{
  CudaContext()
    : s(new detail::CudaStream{})
  {
    inc_ref();
  }

  CudaContext(CudaContext const& other)
    : s(other.s)
  {
    inc_ref();
  }

  CudaContext(CudaContext && other)
    : s(other.s)
  {
    other.s = nullptr;
  }

  CudaContext& operator=(CudaContext const& rhs)
  {
    if (s != rhs.s) {
      dec_ref();
      s = rhs.s;
      inc_ref();
    }
    return *this;
  }

  CudaContext& operator=(CudaContext && rhs)
  {
    if (s != rhs.s) {
      dec_ref();
    }
    s = rhs.s;
    rhs.s = nullptr;
    return *this;
  }

  ~CudaContext()
  {
    if (s != nullptr) {
      dec_ref();
    }
  }

  cudaStream_t stream() const
  {
    return s->stream();
  }

  void waitOn(CPUContext const&) const
  {
    // do nothing
  }

  void waitOn(MPIContext const&) const
  {
    // do nothing
  }

  void waitOn(CudaContext const& other) const
  {
    if (s != other.s) {
      s->waitOn(*other.s);
    }
  }

  void synchronize() const
  {
    s->synchronize();
  }

private:
  detail::CudaStream* s;

  void inc_ref()
  {
    s->inc();
  }

  void dec_ref()
  {
    if (s->dec() == 0) {
      delete s;
      s = nullptr;
    }
  }
};

#endif

struct CPUContext
{
  void waitOn(CPUContext const&) const
  {
    // do nothing
  }

  void waitOn(MPIContext const&) const
  {
    // do nothing
  }

#ifdef COMB_ENABLE_CUDA
  void waitOn(CudaContext const& other) const
  {
    other.synchronize();
  }
#endif

  void synchronize() const
  {
    // do nothing
  }
};

struct MPIContext
{
  void waitOn(CPUContext const&) const
  {
    // do nothing
  }

  void waitOn(MPIContext const&) const
  {
    // do nothing
  }

#ifdef COMB_ENABLE_CUDA
  void waitOn(CudaContext const& other) const
  {
    other.synchronize();
  }
#endif

  void synchronize() const
  {
    // do nothing
  }
};


template < typename exec_pol >
struct ExecContext;

template < typename comm_pol >
struct CommContext;

#endif // _EXECCONTEXT_HPP
