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

#include "cuda_utils.hpp"

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

struct Allocator
{
  virtual const char* name() = 0;
  virtual void* allocate(size_t) = 0;
  virtual void deallocate(void*) = 0;
};

struct HostAllocator : Allocator
{
  virtual const char* name() { return "Host"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = malloc(nbytes);
    // FPRINTF(stdout, "allocated %p nbytes %zu\n", ptr, nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    // FPRINTF(stdout, "deallocating %p\n", ptr);
    free(ptr);
  }
};

#ifdef COMB_ENABLE_CUDA
struct HostPinnedAllocator : Allocator
{
  virtual const char* name() { return "HostPinned"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_host_pinned_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_host_pinned_allocator>::getInstance().free(ptr);
  }
};

struct DeviceAllocator : Allocator
{
  virtual const char* name() { return "Device"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_device_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_device_allocator>::getInstance().free(ptr);
  }
};

struct ManagedAllocator : Allocator
{
  virtual const char* name() { return "Managed"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_allocator>::getInstance().free(ptr);
  }
};

struct ManagedHostPreferredAllocator : Allocator
{
  virtual const char* name() { return "ManagedHostPreferred"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_host_preferred_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_host_preferred_allocator>::getInstance().free(ptr);
  }
};

struct ManagedHostPreferredDeviceAccessedAllocator : Allocator
{
  virtual const char* name() { return "ManagedHostPreferredDeviceAccessed"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_host_preferred_device_accessed_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_host_preferred_device_accessed_allocator>::getInstance().free(ptr);
  }
};

struct ManagedDevicePreferredAllocator : Allocator
{
  virtual const char* name() { return "ManagedDevicePreferred"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_device_preferred_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_device_preferred_allocator>::getInstance().free(ptr);
  }
};

struct ManagedDevicePreferredHostAccessedAllocator : Allocator
{
  virtual const char* name() { return "ManagedDevicePreferredHostAccessed"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_device_preferred_host_accessed_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_device_preferred_host_accessed_allocator>::getInstance().free(ptr);
  }
};
#endif

#endif // _MEMORY_HPP

