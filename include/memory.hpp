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

#ifndef _MEMORY_HPP
#define _MEMORY_HPP

#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "basic_mempool.hpp"

#include "ExecContext.hpp"
#include "utils_cuda.hpp"

namespace COMB {

namespace detail {

template < typename alloc >
using mempool = RAJA::basic_mempool::MemPool<alloc>;

#ifdef COMB_ENABLE_CUDA
  struct cuda_host_pinned_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaHostAlloc(&ptr, nbytes, cudaHostAllocDefault));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFreeHost(ptr));
    }
  };

  struct cuda_device_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMalloc(&ptr, nbytes));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_read_mostly_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetReadMostly, 0));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_host_preferred_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_host_preferred_device_accessed_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetAccessedBy, cuda::get_device()));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_device_preferred_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, cuda::get_device()));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_device_preferred_host_accessed_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, cuda::get_device()));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };
#endif

} // end detail

struct Allocator
{
  virtual const char* name() { return "Null"; }
  virtual void* allocate(size_t nbytes)
  {
    COMB::ignore_unused(nbytes);
    void* ptr = nullptr;
    // FPRINTF(stdout, "allocated %p nbytes %zu\n", ptr, nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    // FPRINTF(stdout, "deallocating %p\n", ptr);
    assert(ptr == nullptr);
  }
};

struct HostAllocator : Allocator
{
  const char* name() override { return "Host"; }
  void* allocate(size_t nbytes) override
  {
    void* ptr = malloc(nbytes);
    // FPRINTF(stdout, "allocated %p nbytes %zu\n", ptr, nbytes);
    return ptr;
  }
  void deallocate(void* ptr) override
  {
    // FPRINTF(stdout, "deallocating %p\n", ptr);
    free(ptr);
  }
};

struct HostPinnedAllocator : Allocator
{
#ifdef COMB_ENABLE_CUDA
  const char* name() override { return "HostPinned"; }
  void* allocate(size_t nbytes) override
  {
    void* ptr = detail::mempool<detail::cuda_host_pinned_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  void deallocate(void* ptr) override
  {
    detail::mempool<detail::cuda_host_pinned_allocator>::getInstance().free(ptr);
  }
#endif
};

struct DeviceAllocator : Allocator
{
#ifdef COMB_ENABLE_CUDA
  const char* name() override { return "Device"; }
  void* allocate(size_t nbytes) override
  {
    void* ptr = detail::mempool<detail::cuda_device_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  void deallocate(void* ptr) override
  {
    detail::mempool<detail::cuda_device_allocator>::getInstance().free(ptr);
  }
#endif
};

struct ManagedAllocator : Allocator
{
#ifdef COMB_ENABLE_CUDA
  const char* name() override { return "Managed"; }
  void* allocate(size_t nbytes) override
  {
    void* ptr = detail::mempool<detail::cuda_managed_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  void deallocate(void* ptr) override
  {
    detail::mempool<detail::cuda_managed_allocator>::getInstance().free(ptr);
  }
#endif
};

struct ManagedHostPreferredAllocator : Allocator
{
#ifdef COMB_ENABLE_CUDA
  const char* name() override { return "ManagedHostPreferred"; }
  void* allocate(size_t nbytes) override
  {
    void* ptr = detail::mempool<detail::cuda_managed_host_preferred_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  void deallocate(void* ptr) override
  {
    detail::mempool<detail::cuda_managed_host_preferred_allocator>::getInstance().free(ptr);
  }
#endif
};

struct ManagedHostPreferredDeviceAccessedAllocator : Allocator
{
#ifdef COMB_ENABLE_CUDA
  const char* name() override { return "ManagedHostPreferredDeviceAccessed"; }
  void* allocate(size_t nbytes) override
  {
    void* ptr = detail::mempool<detail::cuda_managed_host_preferred_device_accessed_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  void deallocate(void* ptr) override
  {
    detail::mempool<detail::cuda_managed_host_preferred_device_accessed_allocator>::getInstance().free(ptr);
  }
#endif
};

struct ManagedDevicePreferredAllocator : Allocator
{
#ifdef COMB_ENABLE_CUDA
  const char* name() override { return "ManagedDevicePreferred"; }
  void* allocate(size_t nbytes) override
  {
    void* ptr = detail::mempool<detail::cuda_managed_device_preferred_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  void deallocate(void* ptr) override
  {
    detail::mempool<detail::cuda_managed_device_preferred_allocator>::getInstance().free(ptr);
  }
#endif
};

struct ManagedDevicePreferredHostAccessedAllocator : Allocator
{
#ifdef COMB_ENABLE_CUDA
  const char* name() override { return "ManagedDevicePreferredHostAccessed"; }
  void* allocate(size_t nbytes) override
  {
    void* ptr = detail::mempool<detail::cuda_managed_device_preferred_host_accessed_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  void deallocate(void* ptr) override
  {
    detail::mempool<detail::cuda_managed_device_preferred_host_accessed_allocator>::getInstance().free(ptr);
  }
#endif
};

struct AllocatorAccessibilityFlags
{
  // special flag to enable tests that access host pageable memory from the device
  bool cuda_host_accessible_from_device = false;
  // special flag to enable tests that access device memory from the host
  bool cuda_device_accessible_from_host = false;
  // special flag to enable tests that pass device buffers to MPI
  bool cuda_aware_mpi = false;
};

struct AllocatorInfo
{
  bool m_available = false;
  AllocatorInfo(AllocatorAccessibilityFlags& a) : m_accessFlags(a) { }
  virtual Allocator& allocator() = 0;
  virtual bool available() = 0;
  virtual bool accessible(CPUContext const&) = 0;
  virtual bool accessible(MPIContext const&) = 0;
  virtual bool accessible(CudaContext const&) = 0;
protected:
  AllocatorAccessibilityFlags& m_accessFlags;
};

struct InvalidAllocatorInfo : AllocatorInfo
{
  InvalidAllocatorInfo(AllocatorAccessibilityFlags& a) : AllocatorInfo(a) { }
  Allocator& allocator() override { throw std::invalid_argument("InvalidAllocatorInfo has no allocator"); }
  bool available() override { return false; }
  bool accessible(CPUContext const&) override { return false; }
  bool accessible(MPIContext const&) override { return false; }
  bool accessible(CudaContext const&) override { return false; }
};

struct HostAllocatorInfo : AllocatorInfo
{
  HostAllocatorInfo(AllocatorAccessibilityFlags& a) : AllocatorInfo(a) { }
  Allocator& allocator() override { return m_allocator; }
  bool available() override { return m_available; }
  bool accessible(CPUContext const&) override { return true; }
  bool accessible(MPIContext const&) override { return true; }
  bool accessible(CudaContext const&) override { return m_accessFlags.cuda_host_accessible_from_device; }
private:
  HostAllocator m_allocator;
};

struct HostPinnedAllocatorInfo : AllocatorInfo
{
  HostPinnedAllocatorInfo(AllocatorAccessibilityFlags& a) : AllocatorInfo(a) { }
  Allocator& allocator() override { return m_allocator; }
  bool available() override { return m_available; }
  bool accessible(CPUContext const&) override { return true; }
  bool accessible(MPIContext const&) override { return true; }
  bool accessible(CudaContext const&) override { return true; }
private:
  HostPinnedAllocator m_allocator;
};

struct DeviceAllocatorInfo : AllocatorInfo
{
  DeviceAllocatorInfo(AllocatorAccessibilityFlags& a) : AllocatorInfo(a) { }
  Allocator& allocator() override { return m_allocator; }
  bool available() override { return m_available; }
  bool accessible(CPUContext const&) override { return m_accessFlags.cuda_device_accessible_from_host; }
  bool accessible(MPIContext const&) override { return m_accessFlags.cuda_aware_mpi; }
  bool accessible(CudaContext const&) override { return true; }
private:
  DeviceAllocator m_allocator;
};

struct ManagedAllocatorInfo : AllocatorInfo
{
  ManagedAllocatorInfo(AllocatorAccessibilityFlags& a) : AllocatorInfo(a) { }
  Allocator& allocator() override { return m_allocator; }
  bool available() override { return m_available; }
  bool accessible(CPUContext const&) override { return true; }
  bool accessible(MPIContext const&) override { return m_accessFlags.cuda_aware_mpi; }
  bool accessible(CudaContext const&) override { return true; }
private:
  ManagedAllocator m_allocator;
};

struct ManagedHostPreferredAllocatorInfo : AllocatorInfo
{
  ManagedHostPreferredAllocatorInfo(AllocatorAccessibilityFlags& a) : AllocatorInfo(a) { }
  Allocator& allocator() override { return m_allocator; }
  bool available() override { return m_available; }
  bool accessible(CPUContext const&) override { return true; }
  bool accessible(MPIContext const&) override { return m_accessFlags.cuda_aware_mpi; }
  bool accessible(CudaContext const&) override { return true; }
private:
  ManagedHostPreferredAllocator m_allocator;
};

struct ManagedHostPreferredDeviceAccessedAllocatorInfo : AllocatorInfo
{
  ManagedHostPreferredDeviceAccessedAllocatorInfo(AllocatorAccessibilityFlags& a) : AllocatorInfo(a) { }
  Allocator& allocator() override { return m_allocator; }
  bool available() override { return m_available; }
  bool accessible(CPUContext const&) override { return true; }
  bool accessible(MPIContext const&) override { return m_accessFlags.cuda_aware_mpi; }
  bool accessible(CudaContext const&) override { return true; }
private:
  ManagedHostPreferredDeviceAccessedAllocator m_allocator;
};

struct ManagedDevicePreferredAllocatorInfo : AllocatorInfo
{
  ManagedDevicePreferredAllocatorInfo(AllocatorAccessibilityFlags& a) : AllocatorInfo(a) { }
  Allocator& allocator() override { return m_allocator; }
  bool available() override { return m_available; }
  bool accessible(CPUContext const&) override { return true; }
  bool accessible(MPIContext const&) override { return m_accessFlags.cuda_aware_mpi; }
  bool accessible(CudaContext const&) override { return true; }
private:
  ManagedDevicePreferredAllocator m_allocator;
};

struct ManagedDevicePreferredHostAccessedAllocatorInfo : AllocatorInfo
{
  ManagedDevicePreferredHostAccessedAllocatorInfo(AllocatorAccessibilityFlags& a) : AllocatorInfo(a) { }
  Allocator& allocator() override { return m_allocator; }
  bool available() override { return m_available; }
  bool accessible(CPUContext const&) override { return true; }
  bool accessible(MPIContext const&) override { return m_accessFlags.cuda_aware_mpi; }
  bool accessible(CudaContext const&) override { return true; }
private:
  ManagedDevicePreferredHostAccessedAllocator m_allocator;
};

struct Allocators
{
  AllocatorAccessibilityFlags access;

  InvalidAllocatorInfo                            invalid{access};
  HostAllocatorInfo                               host{access};
  HostPinnedAllocatorInfo                         cuda_hostpinned{access};
  DeviceAllocatorInfo                             cuda_device{access};
  ManagedAllocatorInfo                            cuda_managed{access};
  ManagedHostPreferredAllocatorInfo               cuda_managed_host_preferred{access};
  ManagedHostPreferredDeviceAccessedAllocatorInfo cuda_managed_host_preferred_device_accessed{access};
  ManagedDevicePreferredAllocatorInfo             cuda_managed_device_preferred{access};
  ManagedDevicePreferredHostAccessedAllocatorInfo cuda_managed_device_preferred_host_accessed{access};
};

} // namespace COMB

#endif // _MEMORY_HPP

