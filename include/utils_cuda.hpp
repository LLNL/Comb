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

#ifndef _CUDA_UTILS_HPP
#define _CUDA_UTILS_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_CUDA

#include <cassert>
#include <cstdio>

#include <cuda.h>
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
// #include <mpi.h>

namespace COMB {

#define IS_DEVICE_LAMBDA(kernel_typetype) \
        __nv_is_extended_device_lambda_closure_type(kernel_type) || \
        __nv_is_extended_host_device_lambda_closure_type(kernel_type)

#define cudaCheck(...) ::COMB::cudaCheckError(#__VA_ARGS__, __VA_ARGS__, __FILE__, __LINE__)

inline void cudaCheckError(const char* str, cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "Error performing %s; %s %s %s:%i\n", str, cudaGetErrorName(code), cudaGetErrorString(code), file, line); fflush(stderr);
    assert(0);
    // MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

#define cudaCheckReady(...) ::COMB::cudaCheckReadyError(#__VA_ARGS__, __VA_ARGS__, __FILE__, __LINE__)

inline bool cudaCheckReadyError(const char* str, cudaError_t code, const char* file, int line)
{
  if (code == cudaSuccess) {
    return true;
  } else if (code != cudaErrorNotReady) {
    fprintf(stderr, "Error performing %s; %s %s %s:%i\n", str, cudaGetErrorName(code), cudaGetErrorString(code), file, line); fflush(stderr);
    assert(0);
    // MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return false;
}


namespace detail {

namespace cuda {

inline int get_device_impl() {
  int d = -1;
  cudaCheck(cudaGetDevice(&d));
  return d;
}

inline int get_device() {
  static int d = get_device_impl();
  return d;
}

inline cudaDeviceProp get_properties_impl() {
  cudaDeviceProp p;
  cudaCheck(cudaGetDeviceProperties(&p, get_device()));
  return p;
}

inline cudaDeviceProp get_properties() {
  static cudaDeviceProp p = get_properties_impl();
  return p;
}

inline int get_concurrent_managed_access() {
  static int accessible =
#if defined(CUDART_VERSION) && CUDART_VERSION >= 8000
    get_properties().concurrentManagedAccess;
#else
    false;
#endif
  return accessible;
}

inline int get_host_accessible_from_device() {
  static int accessible =
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
    get_properties().pageableMemoryAccess;
#else
    false;
#endif
  return accessible;
}

inline int get_device_accessible_from_host() {
  static int accessible =
    false;
  return accessible;
}

inline int get_num_sm() {
  static int num_sm = get_properties().multiProcessorCount;
  return num_sm;
}

inline int get_arch() {
  static int cuda_arch = 100*get_properties().major + 10*get_properties().minor;
  return cuda_arch;
}

__device__ __forceinline__ unsigned long long device_timer()
{
  unsigned long long global_timer = 0;
#if __CUDA_ARCH__ >= 300
  asm volatile ("mov.u64 %0, %globaltimer;" : "=l"(global_timer));
#endif
  return global_timer;
}

} // namespace cuda

} // namespace detail

} // namespace COMB

#endif

#endif // _CUDA_UTILS_HPP
