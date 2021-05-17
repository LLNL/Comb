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

#ifndef _EXECCONTEXT_HPP
#define _EXECCONTEXT_HPP

#include "config.hpp"

#include "print.hpp"
#include "exec_utils_cuda.hpp"

#ifdef COMB_ENABLE_RAJA
#include "RAJA/RAJA.hpp"
#endif

enum struct ContextEnum
{
  invalid = 0
 ,cpu = 1
 ,cuda = 2
 ,mpi = 3
 ,raja_host = 4
 ,raja_cuda = 5
};

struct CPUContext;

#ifdef COMB_ENABLE_MPI
struct MPIContext;
#endif

#ifdef COMB_ENABLE_CUDA
struct CudaContext;
#endif

#ifdef COMB_ENABLE_RAJA
template < typename Resource >
struct RAJAContext;
#endif

namespace detail {

#ifdef COMB_ENABLE_CUDA

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

  void waitEvent(cudaEvent_t other_event)
  {
    cudaCheck(cudaStreamWaitEvent(m_stream, other_event, 0));
    m_record_current = false;
    m_sync = false;
  }

  void waitEvent(CudaStream& other)
  {
    waitEvent(other.m_event);
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

#endif

} // namesapce detail


struct CPUContext
{
  CPUContext()
  {
    LOGPRINTF("%p CPUContext::CPUContext\n", this);
  }

  CPUContext(CPUContext const& other)
  {
    LOGPRINTF("%p CPUContext::CPUContext CPUContext %p\n", this, &other);
  }

  CPUContext(CPUContext && other) noexcept
  {
    LOGPRINTF("%p CPUContext::CPUContext CPUContext&& %p\n", this, &other);
  }

  CPUContext& operator=(CPUContext const& rhs)
  {
    LOGPRINTF("%p CPUContext::operator= CPUContext %p\n", this, &rhs);
    return *this;
  }

  CPUContext& operator=(CPUContext && rhs)
  {
    LOGPRINTF("%p CPUContext::operator= CPUContext&& %p\n", this, &rhs);
    return *this;
  }

  ~CPUContext()
  {
    LOGPRINTF("%p CPUContext::~CPUContext\n", this);
  }

  void waitOn(CPUContext& other)
  {
    LOGPRINTF("%p CPUContext::waitOn CPUContext %p\n", this, &other);
    // do nothing
  }

#ifdef COMB_ENABLE_MPI
  inline void waitOn(MPIContext& other);
#endif

#ifdef COMB_ENABLE_CUDA
  inline void waitOn(CudaContext& other);
#endif

#ifdef COMB_ENABLE_RAJA
  template < typename Resource >
  inline void waitOn(RAJAContext<Resource>& other);
#endif

  void synchronize()
  {
    LOGPRINTF("%p CPUContext::synchronize\n", this);
    // do nothing
  }
};

#ifdef COMB_ENABLE_MPI

struct MPIContext
{
  MPIContext()
  {
    LOGPRINTF("%p MPIContext::MPIContext\n", this);
  }

  MPIContext(MPIContext const& other)
  {
    LOGPRINTF("%p MPIContext::MPIContext MPIContext %p\n", this, &other);
  }

  MPIContext(MPIContext && other) noexcept
  {
    LOGPRINTF("%p MPIContext::MPIContext MPIContext&& %p\n", this, &other);
  }

  MPIContext& operator=(MPIContext const& rhs)
  {
    LOGPRINTF("%p MPIContext::operator= MPIContext %p\n", this, &rhs);
    return *this;
  }

  MPIContext& operator=(MPIContext && rhs)
  {
    LOGPRINTF("%p MPIContext::operator= MPIContext&& %p\n", this, &rhs);
    return *this;
  }

  ~MPIContext()
  {
    LOGPRINTF("%p MPIContext::~MPIContext\n", this);
  }

  inline void waitOn(CPUContext& other);

  void waitOn(MPIContext& other)
  {
    LOGPRINTF("%p MPIContext::waitOn MPIContext %p\n", this, &other);
    // do nothing
  }

#ifdef COMB_ENABLE_CUDA
  inline void waitOn(CudaContext& other);
#endif

#ifdef COMB_ENABLE_RAJA
  template < typename Resource >
  inline void waitOn(RAJAContext<Resource>& other);
#endif

  void synchronize()
  {
    LOGPRINTF("%p MPIContext::synchronize\n", this);
    // do nothing
  }
};

#endif


#ifdef COMB_ENABLE_CUDA

struct CudaContext
{
  CudaContext()
    : s(new detail::CudaStream{})
  {
    LOGPRINTF("%p CudaContext::CudaContext\n", this);
    inc_ref();
  }

  CudaContext(CudaContext const& other)
    : s(other.s)
  {
    LOGPRINTF("%p CudaContext::CudaContext CudaContext %p\n", this, &other);
    inc_ref();
  }

  CudaContext(CudaContext && other) noexcept
    : s(other.s)
  {
    LOGPRINTF("%p CudaContext::CudaContext CudaContext&& %p\n", this, &other);
    other.s = nullptr;
  }

  CudaContext& operator=(CudaContext const& rhs)
  {
    LOGPRINTF("%p CudaContext::operator= CudaContext %p\n", this, &rhs);
    if (s != rhs.s) {
      dec_ref();
      s = rhs.s;
      inc_ref();
    }
    return *this;
  }

  CudaContext& operator=(CudaContext && rhs)
  {
    LOGPRINTF("%p CudaContext::operator= CudaContext&& %p\n", this, &rhs);
    if (s != rhs.s) {
      dec_ref();
    }
    s = rhs.s;
    rhs.s = nullptr;
    return *this;
  }

  ~CudaContext()
  {
    LOGPRINTF("%p CudaContext::~CudaContext\n", this);
    dec_ref();
  }

  cudaStream_t stream()
  {
    return s->stream();
  }

  cudaStream_t stream_launch()
  {
    return s->stream_launch();
  }

  inline void waitOn(CPUContext& other);

#ifdef COMB_ENABLE_MPI
  inline void waitOn(MPIContext& other);
#endif

  void waitOn(CudaContext& other)
  {
    LOGPRINTF("%p CudaContext::waitOn CudaContext %p\n", this, &other);
    assert(s != nullptr);
    assert(other.s != nullptr);
    if (s != other.s) {
      s->waitOn(*other.s);
    }
  }

#ifdef COMB_ENABLE_RAJA
  template < typename Resource >
  inline void waitOn(RAJAContext<Resource>& other);
#endif

  void synchronize()
  {
    LOGPRINTF("%p CudaContext::synchronize\n", this);
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


#ifdef COMB_ENABLE_RAJA

template < typename Resource >
struct RAJAContext
{
  RAJAContext()
  {
    LOGPRINTF("%p RAJAContext::RAJAContext\n", this);
  }

  RAJAContext(RAJAContext const& other)
    : r(other.r)
  {
    LOGPRINTF("%p RAJAContext::RAJAContext RAJAContext %p\n", this, &other);
  }

  RAJAContext(RAJAContext && other) noexcept
    : r(std::move(other.r))
  {
    LOGPRINTF("%p RAJAContext::RAJAContext RAJAContext&& %p\n", this, &other);
  }

  RAJAContext& operator=(RAJAContext const& rhs)
  {
    LOGPRINTF("%p RAJAContext::operator= RAJAContext %p\n", this, &rhs);
    if (this != &rhs) {
      r = rhs.r;
    }
    return *this;
  }

  RAJAContext& operator=(RAJAContext && rhs)
  {
    LOGPRINTF("%p RAJAContext::operator= RAJAContext&& %p\n", this, &rhs);
    if (this != &rhs) {
      r = std::move(rhs.r);
    }
    return *this;
  }

  ~RAJAContext()
  {
    LOGPRINTF("%p RAJAContext::~RAJAContext\n", this);
  }

  Resource& resource()
  {
    return r;
  }

  Resource& res_launch()
  {
    return r;
  }

  inline void waitOn(CPUContext& other);

#ifdef COMB_ENABLE_MPI
  inline void waitOn(MPIContext& other);
#endif

#ifdef COMB_ENABLE_CUDA
  inline void waitOn(CudaContext& other);
#endif

  template < typename OResource >
  void waitOn(RAJAContext<OResource>& other)
  {
    LOGPRINTF("%p RAJAContext::waitOn RAJAContext %p\n", this, &other);
    RAJA::resources::Event e = other.resource().get_event_erased();
    r.wait_for(&e);
  }

  void synchronize()
  {
    LOGPRINTF("%p RAJAContext::synchronize\n", this);
    r.wait();
  }

private:
  // TODO: change this from default
  Resource r = Resource::get_default();
};

#endif


#ifdef COMB_ENABLE_MPI
inline void CPUContext::waitOn(MPIContext& other)
{
  LOGPRINTF("%p CPUContext::waitOn MPIContext %p\n", this, &other);
  // do nothing
}
#endif

#ifdef COMB_ENABLE_CUDA
inline void CPUContext::waitOn(CudaContext& other)
{
  LOGPRINTF("%p CPUContext::waitOn CudaContext %p\n", this, &other);
  other.synchronize();
}
#endif

#ifdef COMB_ENABLE_RAJA
template < typename Resource >
inline void CPUContext::waitOn(RAJAContext<Resource>& other)
{
  LOGPRINTF("%p CPUContext::waitOn RAJAContext %p\n", this, &other);
  other.synchronize();
}
#endif


#ifdef COMB_ENABLE_MPI

inline void MPIContext::waitOn(CPUContext& other)
{
  LOGPRINTF("%p MPIContext::waitOn CPUContext %p\n", this, &other);
  // do nothing
}

#ifdef COMB_ENABLE_CUDA
inline void MPIContext::waitOn(CudaContext& other)
{
  LOGPRINTF("%p MPIContext::waitOn CudaContext %p\n", this, &other);
  other.synchronize();
}
#endif

#ifdef COMB_ENABLE_RAJA
template < typename Resource >
inline void MPIContext::waitOn(RAJAContext<Resource>& other)
{
  LOGPRINTF("%p MPIContext::waitOn RAJAContext %p\n", this, &other);
  other.synchronize();
}
#endif

#endif


#ifdef COMB_ENABLE_CUDA

inline void CudaContext::waitOn(CPUContext& other)
{
  LOGPRINTF("%p CudaContext::waitOn CPUContext %p\n", this, &other);
  // do nothing
}

#ifdef COMB_ENABLE_MPI
inline void CudaContext::waitOn(MPIContext& other)
{
  LOGPRINTF("%p CudaContext::waitOn MPIContext %p\n", this, &other);
  // do nothing
}
#endif

#ifdef COMB_ENABLE_RAJA
template < typename Resource >
inline void CudaContext::waitOn(RAJAContext<Resource>& other)
{
  LOGPRINTF("%p CudaContext::waitOn RAJAContext %p\n", this, &other);
  other.synchronize();
}
#ifdef COMB_ENABLE_CUDA
template < >
inline void CudaContext::waitOn(RAJAContext<RAJA::resources::Cuda>& other)
{
  LOGPRINTF("%p CudaContext::waitOn RAJAContext %p\n", this, &other);
  auto e = other.resource().get_event();
  s->waitEvent(e.getCudaEvent_t());
}
#endif
#endif

#endif


#ifdef COMB_ENABLE_RAJA

template < typename Resource >
inline void RAJAContext<Resource>::waitOn(CPUContext& other)
{
  LOGPRINTF("%p RAJAContext::waitOn CPUContext %p\n", this, &other);
  // do nothing
}

#ifdef COMB_ENABLE_MPI
template < typename Resource >
inline void RAJAContext<Resource>::waitOn(MPIContext& other)
{
  LOGPRINTF("%p RAJAContext::waitOn MPIContext %p\n", this, &other);
  // do nothing
}
#endif

#ifdef COMB_ENABLE_CUDA
template < typename Resource >
inline void RAJAContext<Resource>::waitOn(CudaContext& other)
{
  LOGPRINTF("%p RAJAContext::waitOn CudaContext %p\n", this, &other);
  other.synchronize();
}
///
template < >
inline void RAJAContext<RAJA::resources::Cuda>::waitOn(CudaContext& other)
{
  LOGPRINTF("%p RAJAContext::waitOn CudaContext %p\n", this, &other);
  RAJA::resources::Event e(RAJA::resources::CudaEvent(other.stream()));
  r.wait_for(&e);
}
#endif

#endif


template < typename exec_pol >
struct ExecContext;

template < typename comm_pol >
struct CommContext;

#endif // _EXECCONTEXT_HPP
