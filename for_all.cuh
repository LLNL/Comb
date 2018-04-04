
#ifndef _FOR_ALL_CUH
#define _FOR_ALL_CUH

#include <cstdio>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda.h>

#include <type_traits>

#include "utils.cuh"
#include "memory.cuh"
#include "batch_launch.cuh"
#include "persistent_launch.cuh"

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
struct cuda_batch_pol {
  static const bool async = true;
  static constexpr const char* name = "cudaBatch";
};
struct cuda_persistent_pol {
  static const bool async = true;
  static constexpr const char* name = "cudaPersistent";
};

// synchronization functions
inline void synchronize(seq_pol const&)
{
}

inline void synchronize(omp_pol const&)
{
}

inline void synchronize(cuda_pol const&)
{
  cudaCheck(cudaDeviceSynchronize());
}

inline void synchronize(cuda_batch_pol const&)
{
  cuda::batch_launch::synchronize();
}

inline void synchronize(cuda_persistent_pol const&)
{
  cuda::persistent_launch::synchronize();
}

namespace detail {

template < typename type >
struct Synchronizer {
  type const& t;
  Synchronizer(type const& t_) : t(t_) {}
  void operator()() const { synchronize(t); }
};

template < typename type, size_t repeats >
struct ConditionalSynchronizer {
  ConditionalSynchronizer(type const&) {}
  void operator()() const { }
};

template < typename type >
struct ConditionalSynchronizer<type, 0> : Synchronizer<type> {
  using parent = Synchronizer<type>;
  ConditionalSynchronizer(type const& t_) : parent(t_) {}
  void operator()() const { parent::operator()(); }
};

template < typename ... types >
struct MultiSynchronizer;

template < >
struct MultiSynchronizer<> {
  void operator()() const { }
};

template < typename type0, typename ... types >
struct MultiSynchronizer<type0, types...> : ConditionalSynchronizer<type0, Count<type0, types...>::value>, MultiSynchronizer<types...> {
  using cparent = ConditionalSynchronizer<type0, Count<type0, types...>::value>;
  using mparent = MultiSynchronizer<types...>;
  MultiSynchronizer(type0 const& t_, types const&... ts_) : cparent(t_), mparent(ts_...) {}
  void operator()() const { cparent::operator()(); mparent::operator()(); }
};

} // namespace detail

template < typename policy0, typename policy1, typename... policies >
inline void synchronize(policy0 const& p0, policy1 const& p1, policies const&...ps)
{
  detail::MultiSynchronizer<policy0, policy1, policies...>{p0, p1, ps...}();
}

namespace detail {

template < typename body_type >
struct adapter_2d {
  IdxT begin0, begin1;
  IdxT len1;
  body_type body;
  template < typename body_type_ >
  adapter_2d(IdxT begin0_, IdxT end0_, IdxT begin1_, IdxT end1_, body_type_&& body_)
    : begin0(begin0_)
    , begin1(begin1_)
    , len1(end1_ - begin1_)
    , body(std::forward<body_type_>(body_))
  { }
  HOST DEVICE
  void operator() (IdxT, IdxT idx) const
  {
    IdxT i0 = idx / len1;
    IdxT i1 = idx - i0 * len1;
    body(i0 + begin0, i1 + begin1, idx);
  }
};

template < typename body_type >
struct adapter_3d {
  IdxT begin0, begin1, begin2;
  IdxT len1, len12;
  body_type body;
  template < typename body_type_ >
  adapter_3d(IdxT begin0_, IdxT end0_, IdxT begin1_, IdxT end1_, IdxT begin2_, IdxT end2_, body_type_&& body_)
    : begin0(begin0_)
    , begin1(begin1_)
    , begin2(begin2_)
    , len1(end1_ - begin1_)
    , len12((end1_ - begin1_) * (end2_ - begin2_))
    , body(std::forward<body_type_>(body_))
  { }
  HOST DEVICE
  void operator() (IdxT, IdxT idx) const
  {
    IdxT i0 = idx / len12;
    IdxT idx12 = idx - i0 * len12;

    IdxT i1 = idx12 / len1;
    IdxT i2 = idx12 - i1 * len1;

    body(i0 + begin0, i1 + begin1, i2 + begin2, idx);
  }
};

} // namespace detail

// for_all functions
template < typename body_type >
inline void for_all(seq_pol const& pol, IdxT begin, IdxT end, body_type&& body)
{
  IdxT i = 0;
  for(IdxT i0 = begin; i0 < end; ++i0) {
    body(i0, i++);
  }
  //synchronize(pol);
}

template < typename body_type >
inline void for_all(omp_pol const& pol, IdxT begin, IdxT end, body_type&& body)
{
  const IdxT len = end - begin;
#pragma omp parallel for
  for(IdxT i = 0; i < len; ++i) {
    body(i + begin, i);
  }
  //synchronize(pol);
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
inline void for_all(cuda_pol const& pol, IdxT begin, IdxT end, body_type&& body)
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
  //synchronize(pol);
}

template < typename body_type >
inline void for_all(cuda_batch_pol const& pol, IdxT begin, IdxT end, body_type&& body)
{
  cuda::batch_launch::for_all(begin, end, std::forward<body_type>(body));
  //synchronize(pol);
}

template < typename body_type >
inline void for_all(cuda_persistent_pol const& pol, IdxT begin, IdxT end, body_type&& body)
{
  cuda::persistent_launch::for_all(begin, end, std::forward<body_type>(body));
  //synchronize(pol);
}


template < typename body_type >
void for_all_2d(seq_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  IdxT i = 0;
  for(IdxT i0 = begin0; i0 < end0; ++i0) {
    for(IdxT i1 = begin1; i1 < end1; ++i1) {
      body(i0, i1, i++);
    }
  }
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_2d(omp_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
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
  //synchronize(pol);
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
inline void for_all_2d(cuda_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
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
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_2d(cuda_batch_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  IdxT len = (end0 - begin0) * (end1 - begin1);
  cuda::batch_launch::for_all(0, len, detail::adapter_2d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, std::forward<body_type>(body)});
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_2d(cuda_persistent_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  IdxT len = (end0 - begin0) * (end1 - begin1);
  cuda::persistent_launch::for_all(0, len, detail::adapter_2d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, std::forward<body_type>(body)});
  //synchronize(pol);
}


template < typename body_type >
inline void for_all_3d(seq_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  IdxT i = 0;
  for(IdxT i0 = begin0; i0 < end0; ++i0) {
    for(IdxT i1 = begin1; i1 < end1; ++i1) {
      for(IdxT i2 = begin2; i2 < end2; ++i2) {
        body(i0, i1, i2, i++);
      }
    }
  }
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_3d(omp_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
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
  //synchronize(pol);
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
inline void for_all_3d(cuda_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
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
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_3d(cuda_batch_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  IdxT len = (end0 - begin0) * (end1 - begin1) * (end2 - begin2);
  cuda::batch_launch::for_all(0, len, detail::adapter_3d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body)});
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_3d(cuda_persistent_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  IdxT len = (end0 - begin0) * (end1 - begin1) * (end2 - begin2);
  cuda::persistent_launch::for_all(0, len, detail::adapter_3d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body)});
  //synchronize(pol);
}

#endif // _FOR_ALL_CUH

