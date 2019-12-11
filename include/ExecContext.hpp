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
 ,mpi = 3
};

struct CPUContext;

#ifdef COMB_ENABLE_MPI
struct MPIContext;
#endif

#ifdef COMB_ENABLE_CUDA

struct CudaContext;

namespace detail {

struct CudaStream
{
  CudaStream()
    : m_ref(0)
    , m_record_current(false)
    , m_sync(true)
  {
    cudaCheck(cudaStreamCreateWithFlags(&m_stream, cudaStreamDefault));
    cudaCheck(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
  }

  ~CudaStream()
  {
    cudaCheck(cudaEventDestroy(m_event));
    cudaCheck(cudaStreamDestroy(m_stream));
  }

  void recordEvent()
  {
    if (!m_record_current) {
      cudaCheck(cudaEventRecord(m_event, m_stream));
      m_record_current = true;
    }
  }

  void waitEvent(CudaStream& other)
  {
    cudaCheck(cudaStreamWaitEvent(m_stream, other.m_event, 0));
    m_record_current = false;
    m_sync = false;
  }

  void waitOn(CudaStream& other)
  {
    other.recordEvent();
    waitEvent(other);
  }

  void synchronize()
  {
    if (!m_sync) {
      cudaCheck(cudaStreamSynchronize(m_stream));
      m_sync = true;
    }
  }

  cudaStream_t stream()
  {
    return m_stream;
  }

  cudaStream_t stream_launch()
  {
    m_record_current = false;
    m_sync = false;
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
  bool m_record_current;
  bool m_sync;
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
    dec_ref();
  }

  cudaStream_t stream() const
  {
    return s->stream();
  }

  cudaStream_t stream_launch() const
  {
    return s->stream_launch();
  }

  void waitOn(CPUContext const&) const
  {
    // do nothing
  }

#ifdef COMB_ENABLE_MPI
  void waitOn(MPIContext const&) const
  {
    // do nothing
  }
#endif

  void waitOn(CudaContext const& other) const
  {
    assert(s != nullptr);
    assert(other.s != nullptr);
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
    if (s != nullptr) {
      s->inc();
    }
  }

  void dec_ref()
  {
    if (s != nullptr && s->dec() == 0) {
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

#ifdef COMB_ENABLE_MPI
  void waitOn(MPIContext const&) const
  {
    // do nothing
  }
#endif

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

#ifdef COMB_ENABLE_MPI

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

#endif


template < typename exec_pol >
struct ExecContext;

template < typename comm_pol >
struct CommContext;

#endif // _EXECCONTEXT_HPP
