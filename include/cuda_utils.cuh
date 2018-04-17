
#ifndef _CUDA_UTILS_CUH
#define _CUDA_UTILS_CUH

#include <cassert>
#include <cstdio>

#include <cuda.h>
#include <mpi.h>

#define HOST __host__
#define DEVICE __device__

#define cudaCheck(...) \
  if (__VA_ARGS__ != cudaSuccess) { \
    fprintf(stderr, "Error performing " #__VA_ARGS__ " %s:%i\n", __FILE__, __LINE__); fflush(stderr); \
    /* assert(0); */ \
    MPI_Abort(MPI_COMM_WORLD, 1); \
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

#endif // _CUDA_UTILS_CUH

