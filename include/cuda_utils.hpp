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

#ifndef _CUDA_UTILS_HPP
#define _CUDA_UTILS_HPP

#ifdef COMB_HAVE_CUDA

#include <cassert>
#include <cstdio>

#include <cuda.h>
// #include <mpi.h>

#define IS_DEVICE_LAMBDA(kernel_typetype) \
        __nv_is_extended_device_lambda_closure_type(kernel_type) || \
        __nv_is_extended_host_device_lambda_closure_type(kernel_type)

#define cudaCheck(...) cudaCheckError(#__VA_ARGS__, __VA_ARGS__, __FILE__, __LINE__)

inline void cudaCheckError(const char* str, cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "Error performing %s; %s %s %s:%i\n", str, cudaGetErrorName(code), cudaGetErrorString(code), file, line); fflush(stderr);
    assert(0);
    // MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

#define cudaCheckReady(...) cudaCheckReadyError(#__VA_ARGS__, __VA_ARGS__, __FILE__, __LINE__)

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

inline int get_device() {
  static int d = -1;
  if (d == -1) {
    cudaCheck(cudaGetDevice(&d));
  }
  return d;
}

inline int get_num_sm() {
   static int num_sm = -1;
   if (num_sm == -1) {
      cudaCheck(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, get_device()));
   }
   return num_sm;
}


inline int get_arch() {
   static int cuda_arch = -1;
   if (cuda_arch == -1) {
      int major, minor;
      cudaCheck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, get_device()));
      cudaCheck(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, get_device()));
      cuda_arch = 100*major + 10*minor;
   }
   return cuda_arch;
}

} // namespace cuda

} // namespace detail


#define HOST __host__
#define DEVICE __device__

#else

#define HOST
#define DEVICE

#endif

#endif // _CUDA_UTILS_HPP
