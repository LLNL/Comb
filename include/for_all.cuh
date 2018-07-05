
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
  static const char* get_name() { return "seq"; }
  using event_type = int;
};
struct omp_pol {
  static const bool async = false;
  static const char* get_name() { return "omp"; }
  using event_type = int;
};
struct cuda_pol {
  static const bool async = true;
  static const char* get_name() { return "cuda"; }
  using event_type = cudaEvent_t;
};
struct cuda_batch_pol {
  static const bool async = true;
  static const char* get_name() { return ( get_batch_always_grid_sync() ? "cudaBatch"      : "cudaBatch_fewgs"      ); }
  using event_type = detail::batch_event_type_ptr;
};
struct cuda_persistent_pol {
  static const bool async = true;
  static const char* get_name() { return ( get_batch_always_grid_sync() ? "cudaPersistent" : "cudaPersistent_fewgs" ); }
  using event_type = detail::batch_event_type_ptr;
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


// force start functions
inline void persistent_launch(seq_pol const&)
{
}

inline void persistent_launch(omp_pol const&)
{
}

inline void persistent_launch(cuda_pol const&)
{
}

inline void persistent_launch(cuda_batch_pol const&)
{
}

inline void persistent_launch(cuda_persistent_pol const&)
{
  cuda::persistent_launch::force_launch();
}


// force complete functions
inline void batch_launch(seq_pol const&)
{
}

inline void batch_launch(omp_pol const&)
{
}

inline void batch_launch(cuda_pol const&)
{
}

inline void batch_launch(cuda_batch_pol const&)
{
  cuda::batch_launch::force_launch();
}

inline void batch_launch(cuda_persistent_pol const&)
{
}


// force complete functions
inline void persistent_stop(seq_pol const&)
{
}

inline void persistent_stop(omp_pol const&)
{
}

inline void persistent_stop(cuda_pol const&)
{
}

inline void persistent_stop(cuda_batch_pol const&)
{
}

inline void persistent_stop(cuda_persistent_pol const&)
{
  cuda::persistent_launch::force_stop();
}


namespace detail {

template < typename type >
struct Synchronizer {
  using value_type = type;
  value_type const& t;
  Synchronizer(value_type const& t_) : t(t_) {}
  void operator()() const { synchronize(t); }
};

template < typename type >
struct PersistentLauncher {
  using value_type = type;
  value_type const& t;
  PersistentLauncher(value_type const& t_) : t(t_) {}
  void operator()() const { persistent_launch(t); }
};

template < typename type >
struct BatchLauncher {
  using value_type = type;
  value_type const& t;
  BatchLauncher(value_type const& t_) : t(t_) {}
  void operator()() const { batch_launch(t); }
};

template < typename type >
struct PersistentStopper {
  using value_type = type;
  value_type const& t;
  PersistentStopper(value_type const& t_) : t(t_) {}
  void operator()() const { persistent_stop(t); }
};

template < typename type, size_t repeats >
struct ConditionalOperator {
  ConditionalOperator(typename type::value_type const&) {}
  void operator()() const { }
};

template < typename type >
struct ConditionalOperator<type, 0> : type {
  using parent = type;
  ConditionalOperator(typename type::value_type const& t_) : parent(t_) {}
  void operator()() const { parent::operator()(); }
};

template < typename ... types >
struct MultiOperator;

template < >
struct MultiOperator<> {
  void operator()() const { }
};

template < typename type0, typename ... types >
struct MultiOperator<type0, types...> : ConditionalOperator<type0, Count<type0, types...>::value>, MultiOperator<types...> {
  using cparent = ConditionalOperator<type0, Count<type0, types...>::value>;
  using mparent = MultiOperator<types...>;
  MultiOperator(typename type0::value_type const& t_, typename types::value_type const&... ts_) : cparent(t_), mparent(ts_...) {}
  void operator()() const { cparent::operator()(); mparent::operator()(); }
};

} // namespace detail

// multiple argument synchronization and other functions
template < typename policy0, typename policy1, typename... policies >
inline void synchronize(policy0 const& p0, policy1 const& p1, policies const&...ps)
{
  detail::MultiOperator<detail::Synchronizer<policy0>, detail::Synchronizer<policy1>, detail::Synchronizer<policies>...>{p0, p1, ps...}();
}

template < typename policy0, typename policy1, typename... policies >
inline void persistent_launch(policy0 const& p0, policy1 const& p1, policies const&...ps)
{
  detail::MultiOperator<detail::PersistentLauncher<policy0>, detail::PersistentLauncher<policy1>, detail::PersistentLauncher<policies>...>{p0, p1, ps...}();
}

template < typename policy0, typename policy1, typename... policies >
inline void batch_launch(policy0 const& p0, policy1 const& p1, policies const&...ps)
{
  detail::MultiOperator<detail::BatchLauncher<policy0>, detail::BatchLauncher<policy1>, detail::BatchLauncher<policies>...>{p0, p1, ps...}();
}

template < typename policy0, typename policy1, typename... policies >
inline void persistent_stop(policy0 const& p0, policy1 const& p1, policies const&...ps)
{
  detail::MultiOperator<detail::PersistentStopper<policy0>, detail::PersistentStopper<policy1>, detail::PersistentStopper<policies>...>{p0, p1, ps...}();
}


// event creation functions
inline typename seq_pol::event_type createEvent(seq_pol const&)
{
  return typename seq_pol::event_type{};
}

inline typename omp_pol::event_type createEvent(omp_pol const&)
{
  return typename omp_pol::event_type{};
}

inline typename cuda_pol::event_type createEvent(cuda_pol const&)
{
  cudaEvent_t event;
  cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  return event;
}

inline typename cuda_batch_pol::event_type createEvent(cuda_batch_pol const&)
{
  return cuda::batch_launch::createEvent();
}

inline typename cuda_persistent_pol::event_type createEvent(cuda_persistent_pol const&)
{
  return cuda::persistent_launch::createEvent();
}


// event record functions
inline void recordEvent(seq_pol const&, typename seq_pol::event_type)
{
}

inline void recordEvent(omp_pol const&, typename omp_pol::event_type)
{
}

inline void recordEvent(cuda_pol const&, typename cuda_pol::event_type event)
{
  cudaCheck(cudaEventRecord(event, cudaStream_t{0}));
}

inline void recordEvent(cuda_batch_pol const&, typename cuda_batch_pol::event_type event)
{
  return cuda::batch_launch::recordEvent(event);
}

inline void recordEvent(cuda_persistent_pol const&, typename cuda_persistent_pol::event_type event)
{
  return cuda::persistent_launch::recordEvent(event);
}


// event query functions
inline bool queryEvent(seq_pol const&, typename seq_pol::event_type)
{
  return true;
}

inline bool queryEvent(omp_pol const&, typename omp_pol::event_type)
{
  return true;
}

inline bool queryEvent(cuda_pol const&, typename cuda_pol::event_type event)
{
  return cudaCheckReady(cudaEventQuery(event));
}

inline bool queryEvent(cuda_batch_pol const&, typename cuda_batch_pol::event_type event)
{
  return cuda::batch_launch::queryEvent(event);
}

inline bool queryEvent(cuda_persistent_pol const&, typename cuda_persistent_pol::event_type event)
{
  return cuda::persistent_launch::queryEvent(event);
}


// event wait functions
inline void waitEvent(seq_pol const&, typename seq_pol::event_type)
{
}

inline void waitEvent(omp_pol const&, typename omp_pol::event_type)
{
}

inline void waitEvent(cuda_pol const&, typename cuda_pol::event_type event)
{
  cudaCheck(cudaEventSynchronize(event));
}

inline void waitEvent(cuda_batch_pol const&, typename cuda_batch_pol::event_type event)
{
  cuda::batch_launch::waitEvent(event);
}

inline void waitEvent(cuda_persistent_pol const&, typename cuda_persistent_pol::event_type event)
{
  cuda::persistent_launch::waitEvent(event);
}


// event destroy functions
inline void destroyEvent(seq_pol const&, typename seq_pol::event_type)
{
}

inline void destroyEvent(omp_pol const&, typename omp_pol::event_type)
{
}

inline void destroyEvent(cuda_pol const&, typename cuda_pol::event_type event)
{
  cudaCheck(cudaEventDestroy(event));
}

inline void destroyEvent(cuda_batch_pol const&, typename cuda_batch_pol::event_type event)
{
  cuda::batch_launch::destroyEvent(event);
}

inline void destroyEvent(cuda_persistent_pol const&, typename cuda_persistent_pol::event_type event)
{
  cuda::persistent_launch::destroyEvent(event);
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

    //FPRINTF(stdout, "adapter_2d (%i+%i %i+%i)%i\n", i0, begin0, i1, begin1, idx);
    //assert(0 <= i0 + begin0 && i0 + begin0 < 3);
    //assert(0 <= i1 + begin1 && i1 + begin1 < 3);

    body(i0 + begin0, i1 + begin1, idx);
  }
};

template < typename body_type >
struct adapter_3d {
  IdxT begin0, begin1, begin2;
  IdxT len2, len12;
  body_type body;
  template < typename body_type_ >
  adapter_3d(IdxT begin0_, IdxT end0_, IdxT begin1_, IdxT end1_, IdxT begin2_, IdxT end2_, body_type_&& body_)
    : begin0(begin0_)
    , begin1(begin1_)
    , begin2(begin2_)
    , len2(end2_ - begin2_)
    , len12((end1_ - begin1_) * (end2_ - begin2_))
    , body(std::forward<body_type_>(body_))
  { }
  HOST DEVICE
  void operator() (IdxT, IdxT idx) const
  {
    IdxT i0 = idx / len12;
    IdxT idx12 = idx - i0 * len12;

    IdxT i1 = idx12 / len2;
    IdxT i2 = idx12 - i1 * len2;

    //FPRINTF(stdout, "adapter_3d (%i+%i %i+%i %i+%i)%i\n", i0, begin0, i1, begin1, i2, begin2, idx);
    //assert(0 <= i0 + begin0 && i0 + begin0 < 3);
    //assert(0 <= i1 + begin1 && i1 + begin1 < 3);
    //assert(0 <= i2 + begin2 && i2 + begin2 < 13);

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

#ifdef COMB_USE_OMP_COLLAPSE

#pragma omp parallel for collapse(2)
  for(IdxT i0 = 0; i0 < len0; ++i0) {
    for(IdxT i1 = 0; i1 < len1; ++i1) {
      IdxT i = i0 * len1 + i1;
      body(i0 + begin0, i1 + begin1, i);
    }
  }

#elif defined(COMB_USE_OMP_WEAK_COLLAPSE)

#pragma omp parallel
  {
    IdxT nthreads = omp_get_num_threads();
    IdxT threadid = omp_get_thread_num();

    const IdxT tmp1 = nthreads / len1;
    const IdxT stride1 = nthreads - tmp1 * len1;

    const IdxT stride0 = tmp1;

    const IdxT tmp11 = threadid / len1;
    IdxT i1 = threadid - tmp11 * len1;

    IdxT i0 = tmp11;

    while (i0 < len0) {

      IdxT i = i0 * len1 + i1;
      body(i0 + begin0, i1 + begin1, i);

      i1 += stride1;

      IdxT carry1 = 0;
      if (i1 >= len1) {
        i1 -= len1;
        carry1 = 1;
      }

      i0 += stride0 + carry1;

    }
  }

#else

#pragma omp parallel for
  for(IdxT i0 = 0; i0 < len0; ++i0) {
    for(IdxT i1 = 0; i1 < len1; ++i1) {
      IdxT i = i0 * len1 + i1;
      body(i0 + begin0, i1 + begin1, i);
    }
  }

#endif
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

#ifdef COMB_USE_OMP_COLLAPSE

#pragma omp parallel for collapse(3)
  for(IdxT i0 = 0; i0 < len0; ++i0) {
    for(IdxT i1 = 0; i1 < len1; ++i1) {
      for(IdxT i2 = 0; i2 < len2; ++i2) {
        IdxT i = i0 * len12 + i1 * len2 + i2;
        body(i0 + begin0, i1 + begin1, i2 + begin2, i);
      }
    }
  }

#elif defined(COMB_USE_OMP_WEAK_COLLAPSE)

#pragma omp parallel
  {
    IdxT nthreads = omp_get_num_threads();
    IdxT threadid = omp_get_thread_num();

    const IdxT tmp2 = nthreads / len2;
    const IdxT stride2 = nthreads - tmp2 * len2;

    const IdxT tmp1 = tmp2 / len1;
    const IdxT stride1 = tmp2 - tmp1 * len1;

    const IdxT stride0 = tmp1;


    const IdxT tmp12 = threadid / len2;
    IdxT i2 = threadid - tmp12 * len2;

    const IdxT tmp11 = tmp12 / len1;
    IdxT i1 = tmp12 - tmp11 * len1;

    IdxT i0 = tmp11;

    while (i0 < len0) {

      IdxT i = i0 * len12 + i1 * len2 + i2;
      body(i0 + begin0, i1 + begin1, i2 + begin2, i);

      i2 += stride2;

      IdxT carry2 = 0;
      if (i2 >= len2) {
        i2 -= len2;
        carry2 = 1;
      }

      i1 += stride1 + carry2;

      IdxT carry1 = 0;
      if (i1 >= len1) {
        i1 -= len1;
        carry1 = 1;
      }

      i0 += stride0 + carry1;

    }
  }

#else

#pragma omp parallel for
  for(IdxT i0 = 0; i0 < len0; ++i0) {
    for(IdxT i1 = 0; i1 < len1; ++i1) {
      for(IdxT i2 = 0; i2 < len2; ++i2) {
        IdxT i = i0 * len12 + i1 * len2 + i2;
        body(i0 + begin0, i1 + begin1, i2 + begin2, i);
      }
    }
  }

#endif
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

