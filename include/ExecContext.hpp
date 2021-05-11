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
  void waitOn(CPUContext const&) const
  {
    // do nothing
  }

#ifdef COMB_ENABLE_MPI
  inline void waitOn(MPIContext const&) const;
#endif

#ifdef COMB_ENABLE_CUDA
  inline void waitOn(CudaContext const& other) const;
#endif

#ifdef COMB_ENABLE_RAJA
  template < typename Resource >
  inline void waitOn(RAJAContext<Resource> const&) const;
#endif

  void synchronize() const
  {
    // do nothing
  }
};

#ifdef COMB_ENABLE_MPI

struct MPIContext
{
  inline void waitOn(CPUContext const&) const;

  void waitOn(MPIContext const&) const
  {
    // do nothing
  }

#ifdef COMB_ENABLE_CUDA
  inline void waitOn(CudaContext const& other) const;
#endif

#ifdef COMB_ENABLE_RAJA
  template < typename Resource >
  inline void waitOn(RAJAContext<Resource> const&) const;
#endif

  void synchronize() const
  {
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
    inc_ref();
  }

  CudaContext(CudaContext const& other)
    : s(other.s)
  {
    inc_ref();
  }

  CudaContext(CudaContext && other) noexcept
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

  inline void waitOn(CPUContext const&) const;

#ifdef COMB_ENABLE_MPI
  inline void waitOn(MPIContext const&) const;
#endif

  void waitOn(CudaContext const& other) const
  {
    assert(s != nullptr);
    assert(other.s != nullptr);
    if (s != other.s) {
      s->waitOn(*other.s);
    }
  }

#ifdef COMB_ENABLE_RAJA
  template < typename Resource >
  inline void waitOn(RAJAContext<Resource> const&) const;
#endif

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


#ifdef COMB_ENABLE_RAJA

template < typename Resource >
struct RAJAContext
{
  Resource& resource()
  {
    return r;
  }

  Resource const& resource() const
  {
    return r;
  }

  Resource& res_launch()
  {
    return r;
  }

  inline void waitOn(CPUContext const&) const;

#ifdef COMB_ENABLE_MPI
  inline void waitOn(MPIContext const&) const;
#endif

#ifdef COMB_ENABLE_CUDA
  inline void waitOn(CudaContext const&) const;
#endif

  template < typename OResource >
  void waitOn(RAJAContext<OResource> const& other) const
  {
    RAJA::resources::Event e = other.resource().get_event_erased();
    auto r_copy = r;
    r_copy.wait_for(&e);
  }

  void synchronize() const
  {
    auto r_copy = r;
    r_copy.wait();
  }

private:
  // TODO: change this from default
  Resource r = Resource::get_default();
};

#endif


#ifdef COMB_ENABLE_MPI
inline void CPUContext::waitOn(MPIContext const&) const
{
  // do nothing
}
#endif

#ifdef COMB_ENABLE_CUDA
inline void CPUContext::waitOn(CudaContext const& other) const
{
  other.synchronize();
}
#endif

#ifdef COMB_ENABLE_RAJA
template < typename Resource >
inline void CPUContext::waitOn(RAJAContext<Resource> const& other) const
{
  other.synchronize();
}
#endif


#ifdef COMB_ENABLE_MPI

inline void MPIContext::waitOn(CPUContext const&) const
{
  // do nothing
}

#ifdef COMB_ENABLE_CUDA
inline void MPIContext::waitOn(CudaContext const& other) const
{
  other.synchronize();
}
#endif

#ifdef COMB_ENABLE_RAJA
template < typename Resource >
inline void MPIContext::waitOn(RAJAContext<Resource> const& other) const
{
  other.synchronize();
}
#endif

#endif


#ifdef COMB_ENABLE_CUDA

inline void CudaContext::waitOn(CPUContext const&) const
{
  // do nothing
}

#ifdef COMB_ENABLE_MPI
inline void CudaContext::waitOn(MPIContext const&) const
{
  // do nothing
}
#endif

#ifdef COMB_ENABLE_RAJA
template < typename Resource >
inline void CudaContext::waitOn(RAJAContext<Resource> const& other) const
{
  other.synchronize();
}
#ifdef COMB_ENABLE_CUDA
template < >
inline void CudaContext::waitOn(RAJAContext<RAJA::resources::Cuda> const& other) const
{
  auto r = other.resource();
  auto e = r.get_event();
  s->waitEvent(e.getCudaEvent_t());
}
#endif
#endif

#endif


#ifdef COMB_ENABLE_RAJA

template < typename Resource >
inline void RAJAContext<Resource>::waitOn(CPUContext const&) const
{
  // do nothing
}

#ifdef COMB_ENABLE_MPI
template < typename Resource >
inline void RAJAContext<Resource>::waitOn(MPIContext const&) const
{
  // do nothing
}
#endif

#ifdef COMB_ENABLE_CUDA
template < typename Resource >
inline void RAJAContext<Resource>::waitOn(CudaContext const& other) const
{
  other.synchronize();
}
///
template < >
inline void RAJAContext<RAJA::resources::Cuda>::waitOn(CudaContext const& other) const
{
  RAJA::resources::Event e(RAJA::resources::CudaEvent(other.stream()));
  auto r_copy = r;
  r_copy.wait_for(&e);
}
#endif

#endif


template < typename exec_pol >
struct ExecContext;

template < typename comm_pol >
struct CommContext;

#endif // _EXECCONTEXT_HPP
