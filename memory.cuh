
#ifndef _MEMORY_CUH
#define _MEMORY_CUH

#include <cstdio>
#include <cstdlib>

#include "utils.cuh"
#include "basic_mempool.hpp"

using IdxT = int;
using LidxT = int;
using DataT = double;

template < typename alloc >
using mempool = RAJA::basic_mempool::MemPool<alloc>;

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

#endif // _MEMORY_CUH

