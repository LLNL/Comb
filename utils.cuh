
#ifndef _UTILS_CUH
#define _UTILS_CUH

#include <cassert>

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

template < typename T, size_t align>
struct aligned_sizeof {
   static const size_t value = sizeof(T) + ((sizeof(T) % align != 0) ? (align - (sizeof(T) % align)) : 0);
};

template < typename T, typename ... types >
struct Count;

template < typename T >
struct Count<T> {
  static const size_t value = 0;
};

template < typename T, typename ... types >
struct Count<T, T, types...> {
  static const size_t value = 1 + Count<T, types...>::value;
};

template < typename T, typename T0, typename ... types >
struct Count<T, T0, types...> {
  static const size_t value = Count<T, types...>::value;
};

} // namespace detail

#endif // _UTILS_CUH

