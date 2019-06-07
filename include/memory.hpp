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

#ifndef _MEMORY_HPP
#define _MEMORY_HPP

#include "config.hpp"

#include <cstdio>
#include <cstdlib>

#include "basic_mempool.hpp"

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
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetAccessedBy, detail::cuda::get_device()));
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
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, detail::cuda::get_device()));
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
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, detail::cuda::get_device()));
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

struct Allocators
{
  HostAllocator host;
  HostPinnedAllocator hostpinned;
  DeviceAllocator device;
  ManagedAllocator managed;
  ManagedHostPreferredAllocator managed_host_preferred;
  ManagedHostPreferredDeviceAccessedAllocator managed_host_preferred_device_accessed;
  ManagedDevicePreferredAllocator managed_device_preferred;
  ManagedDevicePreferredHostAccessedAllocator managed_device_preferred_host_accessed;
};

struct AllocatorsAvailable
{
  bool host = false;
  bool hostpinned = false;
  bool device = false;
  bool managed = false;
  bool managed_host_preferred = false;
  bool managed_host_preferred_device_accessed = false;
  bool managed_device_preferred = false;
  bool managed_device_preferred_host_accessed = false;
  // special flag to enable tests that access host pageable memory from the device
  bool cuda_host_accessible_from_device = false;
};

} // namespace COMB

#endif // _MEMORY_HPP

