
#ifndef _FOR_ALL_CUH
#define _FOR_ALL_CUH

#include <cstdio>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda.h>

#include "memory.cuh"

struct seq_pol {
  static const bool async = false;
  static constexpr const char* name = "seq";
};
struct omp_pol {
  static const bool async = false;
  static constexpr const char* name = "omp";
};
struct cuda_pol {
  static const bool async = true;
  static constexpr const char* name = "cuda";
};

template < typename body_type >
void for_all(seq_pol const&, IdxT begin, IdxT end, body_type&& body)
{
  IdxT i = 0;
  for(IdxT i0 = begin; i0 < end; ++i0) {
    body(i0, i++);
  }
}

template < typename body_type >
void for_all(omp_pol const&, IdxT begin, IdxT end, body_type&& body)
{
  const IdxT len = end - begin;
#pragma omp parallel for
  for(IdxT i = 0; i < len; ++i) {
    body(i + begin, i);
  }
}

template < typename body_type >
__global__
void cuda_for_all(IdxT begin, IdxT len, body_type body)
{
  const IdxT i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < len) {
    body(i + begin, i);
  }
}

template < typename body_type >
void for_all(cuda_pol const&, IdxT begin, IdxT end, body_type&& body)
{
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len = end - begin;

  const IdxT threads = 256;
  const IdxT blocks = (len + threads - 1) / threads;

  void* func = (void*)&cuda_for_all<decayed_body_type>;
  dim3 gridDim(blocks);
  dim3 blockDim(threads);
  void* args[]{&begin, &len, &body};
  size_t sharedMem = 0;
  cudaStream_t stream = 0;
  
  cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
}


template < typename body_type >
void for_all_2d(seq_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  IdxT i = 0;
  for(IdxT i0 = begin0; i0 < end0; ++i0) {
    for(IdxT i1 = begin1; i1 < end1; ++i1) {
      body(i0, i1, i++);
    }
  }
}

template < typename body_type >
void for_all_2d(omp_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  const IdxT len0 = end0 - begin0;
  const IdxT len1 = end1 - begin1;
#pragma omp parallel for collapse(2)
  for(IdxT i0 = 0; i0 < len0; ++i0) {
    for(IdxT i1 = 0; i1 < len1; ++i1) {
      IdxT i = i0 * len1 + i1;
      body(i0 + begin0, i1 + begin1, i);
    }
  }
}

template < typename body_type >
__global__
void cuda_for_all_2d(IdxT begin0, IdxT len0, IdxT begin1, IdxT len1, body_type body)
{
  const IdxT i0 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      IdxT i = i0 * len1 + i1;
      body(i0 + begin0, i1 + begin1, i);
    }
  }
}

template < typename body_type >
void for_all_2d(cuda_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len0 = end0 - begin0;
  IdxT len1 = end1 - begin1;

  const IdxT threads0 = 8;
  const IdxT threads1 = 32;
  const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
  const IdxT blocks1 = (len1 + threads1 - 1) / threads1;

  void* func = (void*)&cuda_for_all_2d<decayed_body_type>;
  dim3 gridDim(blocks1, blocks0, 1);
  dim3 blockDim(threads1, threads0, 1);
  void* args[]{&begin0, &len0, &begin1, &len1, &body};
  size_t sharedMem = 0;
  cudaStream_t stream = 0;
  
  cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
}


template < typename body_type >
void for_all_3d(seq_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  IdxT i = 0;
  for(IdxT i0 = begin0; i0 < end0; ++i0) {
    for(IdxT i1 = begin1; i1 < end1; ++i1) {
      for(IdxT i2 = begin2; i2 < end2; ++i2) {
        body(i0, i1, i2, i++);
      }
    }
  }
}

template < typename body_type >
void for_all_3d(omp_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  const IdxT len0 = end0 - begin0;
  const IdxT len1 = end1 - begin1;
  const IdxT len2 = end2 - begin2;
  const IdxT len12 = len1 * len2;
#pragma omp parallel for collapse(3)
  for(IdxT i0 = 0; i0 < len0; ++i0) {
    for(IdxT i1 = 0; i1 < len1; ++i1) {
      for(IdxT i2 = 0; i2 < len2; ++i2) {
        IdxT i = i0 * len12 + i1 * len2 + i2;
        body(i0 + begin0, i1 + begin1, i2 + begin2, i);
      }
    }
  }
}

template < typename body_type >
__global__
void cuda_for_all_3d(IdxT begin0, IdxT len0, IdxT begin1, IdxT len1, IdxT begin2, IdxT len2, IdxT len12, body_type body)
{
  const IdxT i0 = blockIdx.z;
  const IdxT i1 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i2 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      if (i2 < len2) {
        IdxT i = i0 * len12 + i1 * len2 + i2;
        body(i0 + begin0, i1 + begin1, i2 + begin2, i);
      }
    }
  }
}

template < typename body_type >
void for_all_3d(cuda_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len0 = end0 - begin0;
  IdxT len1 = end1 - begin1;
  IdxT len2 = end2 - begin2;
  IdxT len12 = len1 * len2;

  const IdxT threads0 = 1;
  const IdxT threads1 = 8;
  const IdxT threads2 = 32;
  const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
  const IdxT blocks1 = (len1 + threads1 - 1) / threads1;
  const IdxT blocks2 = (len2 + threads2 - 1) / threads2;

  void* func =(void*)&cuda_for_all_3d<decayed_body_type>;
  dim3 gridDim(blocks2, blocks1, blocks0);
  dim3 blockDim(threads2, threads1, threads0);
  void* args[]{&begin0, &len0, &begin1, &len1, &begin2, &len2, &len12, &body};
  size_t sharedMem = 0;
  cudaStream_t stream = 0;
  
  cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
}

#endif // _FOR_ALL_CUH

