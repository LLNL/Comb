//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2021, Lawrence Livermore National Security, LLC.
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

#ifndef _HIP_UTILS_HPP
#define _HIP_UTILS_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_HIP

#include <cassert>
#include <cstdio>

#include <hip/hip_runtime.h>
#include <roctracer/roctx.h>
// #include <mpi.h>

namespace COMB {

#define IS_DEVICE_LAMBDA(kernel_typetype) \
        __nv_is_extended_device_lambda_closure_type(kernel_type) || \
        __nv_is_extended_host_device_lambda_closure_type(kernel_type)

#define hipCheck(...) ::COMB::hipCheckError(#__VA_ARGS__, __VA_ARGS__, __FILE__, __LINE__)

inline void hipCheckError(const char* str, hipError_t code, const char* file, int line)
{
  if (code != hipSuccess) {
    fprintf(stderr, "Error performing %s; %s %s %s:%i\n", str, hipGetErrorName(code), hipGetErrorString(code), file, line); fflush(stderr);
    assert(0);
    // MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

#define hipCheckReady(...) ::COMB::hipCheckReadyError(#__VA_ARGS__, __VA_ARGS__, __FILE__, __LINE__)

inline bool hipCheckReadyError(const char* str, hipError_t code, const char* file, int line)
{
  if (code == hipSuccess) {
    return true;
  } else if (code != hipErrorNotReady) {
    fprintf(stderr, "Error performing %s; %s %s %s:%i\n", str, hipGetErrorName(code), hipGetErrorString(code), file, line); fflush(stderr);
    assert(0);
    // MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return false;
}


namespace detail {

namespace hip {

inline int get_device_impl() {
  int d = -1;
  hipCheck(hipGetDevice(&d));
  return d;
}

inline int get_device() {
  static int d = get_device_impl();
  return d;
}

inline hipDeviceProp get_properties_impl() {
  hipDeviceProp p;
  hipCheck(hipGetDeviceProperties(&p, get_device()));
  return p;
}

inline hipDeviceProp get_properties() {
  static hipDeviceProp p = get_properties_impl();
  return p;
}

inline int get_concurrent_managed_access() {
  static int accessible =
    true;
  return accessible;
}

inline int get_host_accessible_from_device() {
  static int accessible =
    false;
  return accessible;
}

inline int get_device_accessible_from_host() {
  static int accessible =
    true;
  return accessible;
}

inline int get_num_cu() {
  static int num_cu = get_properties().multiProcessorCount;
  return num_cu;
}

inline int get_arch() {
  static int hip_arch = 100*get_properties().major + 10*get_properties().minor;
  return hip_arch;
}

} // namespace hip

} // namespace detail

} // namespace COMB

#endif

#endif // _HIP_UTILS_HPP
