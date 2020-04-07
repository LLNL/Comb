//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
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

#ifndef _POL_CUDA_HPP
#define _POL_CUDA_HPP

#include "config.hpp"

#include "memory.hpp"

#include <chrono>

#ifdef COMB_ENABLE_CUDA
#include <cuda.h>

#define COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER

#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
#define COMB_CUDA_FUSED_KERNEL_HOST_TIMER
#define COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_SKIP 32u
#define COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_DO_TIMER(i) \
    (((i) % COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_SKIP) == 0)
#define COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_ASSIGN_START(ref, val) \
    ::atomicMin(&(ref), (val))
#define COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_ASSIGN_STOP(ref, val) \
    ::atomicMax(&(ref), (val))
#else
#define COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_DO_TIMER(i) \
    ((i) == 0)
#define COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_ASSIGN_START(ref, val) \
    (ref) = (val)
#define COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_ASSIGN_STOP(ref, val) \
    (ref) = (val)
#endif

#ifdef COMB_CUDA_FUSED_KERNEL_HOST_TIMER
template < int >
__global__
void cuda_timer_kernel(unsigned long long* kernel_timer)
{
  if (kernel_timer != nullptr) {
    kernel_timer[0] = COMB::detail::cuda::device_timer();
  }
}

template < int >
__global__
void cuda_spin_kernel(double time_s)
{
  unsigned long long t0 = COMB::detail::cuda::device_timer();
  unsigned long long time_ns = static_cast<unsigned long long>(time_s * 1000000000.0);
  while (time_ns > COMB::detail::cuda::device_timer()-t0);
}
#endif

template < typename body_type >
__global__
void cuda_for_all(IdxT begin, IdxT len, body_type body
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
  ,unsigned long long* kernel_starts, unsigned long long* kernel_stops, int kernel_id
#endif
  )
{
  const IdxT i = threadIdx.x + blockIdx.x * blockDim.x;
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
  unsigned long long kernel_start;
  if (kernel_starts != nullptr && COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_DO_TIMER(i)) {
    kernel_start = COMB::detail::cuda::device_timer();
  }
#endif
  if (i < len) {
    body(i + begin, i);
  }
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
  if (kernel_starts != nullptr && COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_DO_TIMER(i)) {
    unsigned long long kernel_stop = COMB::detail::cuda::device_timer();
    COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_ASSIGN_START(kernel_starts[kernel_id], kernel_start);
    COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_ASSIGN_STOP(kernel_stops[kernel_id], kernel_stop);
  }
#endif
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

template < typename body_type, IdxT num_threads >
__global__ __launch_bounds__(num_threads)
void cuda_fused(body_type body_in
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
  ,unsigned long long* kernel_starts, unsigned long long* kernel_stops, int kernel_id_begin
#endif
  )
{
  const IdxT i_outer = blockIdx.z;
  const IdxT i_inner = blockIdx.y;
  const IdxT ii      = threadIdx.x + blockIdx.x * blockDim.x;
  const IdxT i_stride = blockDim.x * gridDim.x;
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
  unsigned long long kernel_start;
  if (kernel_starts != nullptr && COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_DO_TIMER(ii)) {
    kernel_start = COMB::detail::cuda::device_timer();
  }
#endif
  auto body = body_in;
  body.set_outer(i_outer);
  body.set_inner(i_inner);
  for (IdxT i = ii; i < body.len; i += i_stride) {
    body(i, i);
  }
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
  if (kernel_starts != nullptr && COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_DO_TIMER(ii)) {
    unsigned long long kernel_stop = COMB::detail::cuda::device_timer();
    COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_ASSIGN_START(kernel_starts[kernel_id_begin + blockIdx.y + blockIdx.z*gridDim.y], kernel_start);
    COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_ASSIGN_STOP(kernel_stops[kernel_id_begin + blockIdx.y + blockIdx.z*gridDim.y], kernel_stop);
  }
#endif
}

struct cuda_component
{
  void* ptr = nullptr;
};

struct cuda_group
{
  struct fused_info
  {
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
    bool m_used = false;
    int m_num_nodes = 0;
    int m_max_num_nodes = 0;
    int m_num_uses = 0;
    unsigned long long* m_kernel_starts = nullptr;
    int m_kernel_id = 0;
    int m_num_kernel_timers = 0;
    bool m_do_timing = false;
#ifdef COMB_CUDA_FUSED_KERNEL_HOST_TIMER
    double m_time_launch = 0.0;
    unsigned long long* m_kernel_pre_start = nullptr;
    cudaEvent_t m_time_launch_events[2];
#endif
#endif
  };
  fused_info* f = nullptr;
};

struct cuda_pol {
  static const bool async = true;
  static const char* get_name() { return "cuda"; }
  using event_type = cudaEvent_t;
  using component_type = cuda_component;
  using group_type = cuda_group;
};

template < >
struct ExecContext<cuda_pol> : CudaContext
{
  using pol = cuda_pol;
  using event_type = typename pol::event_type;
  using component_type = typename pol::component_type;
  using group_type = typename pol::group_type;

  using base = CudaContext;

  COMB::Allocator& util_aloc;


  ExecContext(base const& b, COMB::Allocator& util_aloc_)
    : base(b)
    , util_aloc(util_aloc_)
  { }

  void ensure_waitable()
  {

  }

  template < typename context >
  void waitOn(context& con)
  {
    con.ensure_waitable();
    base::waitOn(con);
  }

  void synchronize()
  {
    base::synchronize();
  }

private:
  static inline group_type& get_active_group()
  {
    static group_type active_group = group_type{};
    return active_group;
  }

public:

  group_type create_group()
  {
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
    group_type g{};
    g.f = new typename group_type::fused_info{};
    return g;
#else
    return group_type{};
#endif
  }

  void start_group(group_type g)
  {
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
    get_active_group() = g;
    assert(g.f);

    g.f->m_used = false;
    g.f->m_num_nodes = 0;
    g.f->m_kernel_id = 0;

    if (g.f->m_num_uses == 5 && g.f->m_kernel_id < g.f->m_max_num_nodes) {
      // enable timing, this should also ensure that m_max_num_nodes > 0
      if (g.f->m_kernel_starts == nullptr) {
        // allocate space to store timers
        g.f->m_num_kernel_timers = g.f->m_max_num_nodes;
        cudaCheck(cudaMalloc(&g.f->m_kernel_starts, (g.f->m_num_kernel_timers*2+2)*sizeof(g.f->m_kernel_starts[0])));
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER_SKIP
        // initialize starts to -1 (ULL_MAX), stops to 0
        cudaCheck(cudaMemsetAsync(g.f->m_kernel_starts, -1, g.f->m_num_kernel_timers*sizeof(g.f->m_kernel_starts[0])));
        cudaCheck(cudaMemsetAsync(g.f->m_kernel_starts+g.f->m_num_kernel_timers, 0, g.f->m_num_kernel_timers*sizeof(g.f->m_kernel_starts[0])));
#endif

#ifdef COMB_CUDA_FUSED_KERNEL_HOST_TIMER
        g.f->m_time_launch = 0.0;
        g.f->m_kernel_pre_start = g.f->m_kernel_starts + g.f->m_num_kernel_timers*2;
        cudaCheck(cudaEventCreateWithFlags(&g.f->m_time_launch_events[0], cudaEventDefault));
        cudaCheck(cudaEventCreateWithFlags(&g.f->m_time_launch_events[1], cudaEventDefault));
#endif
      }
      g.f->m_do_timing = true;
    } else {
      g.f->m_do_timing = false;
    }
#endif
  }

  void finish_group(group_type g)
  {
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
    assert(g.f);
    if (g.f->m_used) {

      if (g.f->m_do_timing) {
        g.f->m_do_timing = false;
      }

      g.f->m_num_uses += 1;
      g.f->m_used = false;
    }

    get_active_group() = group_type{};
#endif
  }

  void destroy_group(group_type g)
  {
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
    assert(g.f);
    if (g.f->m_kernel_starts) {
      unsigned long long* kernel_starts = (unsigned long long*)malloc((g.f->m_num_kernel_timers*2+2)*sizeof(g.f->m_kernel_starts[0]));
      unsigned long long* kernel_stops = kernel_starts + g.f->m_num_kernel_timers;
      cudaCheck(cudaMemcpy(kernel_starts, g.f->m_kernel_starts, (g.f->m_num_kernel_timers*2+2)*sizeof(g.f->m_kernel_starts[0]), cudaMemcpyDefault));
      cudaCheck(cudaFree(g.f->m_kernel_starts));

      double tick_rate = 1000000000.0; // 1 tick / ns

      unsigned long long first_start = kernel_starts[0];
      for (int i = 0; i < g.f->m_num_kernel_timers; i++) {
        if (first_start > kernel_starts[i]) first_start = kernel_starts[i];
      }

#ifdef COMB_CUDA_FUSED_KERNEL_HOST_TIMER
      // print timers from around host api calls
      {
        FGPRINTF(FileGroup::proc, "ExecContext<cuda_pol>(%p).destroy_group fused_launch %.9f\n", this, g.f->m_time_launch);
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, g.f->m_time_launch_events[0], g.f->m_time_launch_events[1]));
        double time = static_cast<double>(ms) / 1000.0;
        FGPRINTF(FileGroup::proc, "ExecContext<cuda_pol>(%p).destroy_group fused_device %.9f\n", this, time);
        cudaCheck(cudaEventDestroy(g.f->m_time_launch_events[0]));
        cudaCheck(cudaEventDestroy(g.f->m_time_launch_events[1]));
        unsigned long long* kernel_pre_start = kernel_starts + g.f->m_num_kernel_timers*2;
        double time_kernel_pre_start = -static_cast<double>(first_start - kernel_pre_start[0]) / tick_rate;
        double time_kernel_post_stop =  static_cast<double>(kernel_pre_start[1] - first_start) / tick_rate;
        FGPRINTF(FileGroup::proc, "ExecContext<cuda_pol>(%p).destroy_group kernel_pre_start %.9f kernel_post_stop %.9f\n", this, time_kernel_pre_start, time_kernel_post_stop);

      }
#endif

      for (int i = 0; i < g.f->m_num_kernel_timers; i++) {
        double start = (kernel_starts[i] - first_start) / tick_rate;
        double stop  = (kernel_stops[i] - first_start) / tick_rate;
        FGPRINTF(FileGroup::proc, "ExecContext<cuda_pol>(%p).destroy_group kernel_start %.9f kernel_stop %.9f\n", this, start, stop);
      }
      free(kernel_starts);
    }
    delete g.f;
#endif
  }

  component_type create_component()
  {
    return component_type{};
  }

  void start_component(group_type, component_type)
  {

  }

  void finish_component(group_type, component_type)
  {

  }

  void destroy_component(component_type)
  {

  }

  event_type createEvent()
  {
    cudaEvent_t event;
    cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
  }

  void recordEvent(event_type event)
  {
    cudaCheck(cudaEventRecord(event, base::stream()));
  }

  void finish_component_recordEvent(group_type group, component_type component, event_type event)
  {
    finish_component(group, component);
    recordEvent(event);
  }

  bool queryEvent(event_type event)
  {
    return cudaCheckReady(cudaEventQuery(event));
  }

  void waitEvent(event_type event)
  {
    cudaCheck(cudaEventSynchronize(event));
  }

  void destroyEvent(event_type event)
  {
    cudaCheck(cudaEventDestroy(event));
  }

  template < typename body_type >
  void for_all(IdxT begin, IdxT end, body_type&& body)
  {
    using decayed_body_type = typename std::decay<body_type>::type;

    IdxT len = end - begin;

    const IdxT threads = 256;
    const IdxT blocks = (len + threads - 1) / threads;

#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
    group_type g = get_active_group();
    unsigned long long* kernel_starts = nullptr;
    unsigned long long* kernel_stops = nullptr;
    int kernel_id = -1;
    if (g.f) {
      if (g.f->m_do_timing) {
        assert(g.f->m_kernel_starts != nullptr);
        assert(g.f->m_kernel_id + 1 <= g.f->m_max_num_nodes);
        kernel_starts = g.f->m_kernel_starts;
        kernel_stops = g.f->m_kernel_starts + g.f->m_num_kernel_timers;
        kernel_id = g.f->m_kernel_id; g.f->m_kernel_id += 1;
      }
      g.f->m_used = true;
      g.f->m_num_nodes += 1;
      if (g.f->m_num_nodes > g.f->m_max_num_nodes) { g.f->m_max_num_nodes = g.f->m_num_nodes; }
    }
#endif

    void* func = (void*)&cuda_for_all<decayed_body_type>;
    dim3 gridDim(blocks);
    dim3 blockDim(threads);
    void* args[]{&begin, &len, &body
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
      , &kernel_starts, &kernel_stops, &kernel_id
#endif
      };
    size_t sharedMem = 0;
    cudaStream_t stream = base::stream_launch();

#ifdef COMB_CUDA_FUSED_KERNEL_HOST_TIMER
    std::chrono::time_point<std::chrono::high_resolution_clock> t0;
    if (g.f && g.f->m_do_timing) {
      if (kernel_id == 0) {
        cuda_spin_kernel<0><<<1,1,0,stream>>>(0.001); // spin for 0.001 seconds
        cuda_timer_kernel<0><<<1,1,0,stream>>>(g.f->m_kernel_pre_start);
        cudaCheck(cudaEventRecord(g.f->m_time_launch_events[0], stream));
      }
      t0 = std::chrono::high_resolution_clock::now();
    }
#endif
    cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
#ifdef COMB_CUDA_FUSED_KERNEL_HOST_TIMER
    if (g.f && g.f->m_do_timing) {
      auto t1 = std::chrono::high_resolution_clock::now();
      if (g.f->m_kernel_id >= g.f->m_max_num_nodes) {
        cudaCheck(cudaEventRecord(g.f->m_time_launch_events[1], stream));
        cuda_timer_kernel<0><<<1,1,0,stream>>>(g.f->m_kernel_pre_start+1);
      }
      g.f->m_time_launch += std::chrono::duration<double>(t1-t0).count();
    }
#endif
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
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
    cudaStream_t stream = base::stream_launch();

    cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
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
    cudaStream_t stream = base::stream_launch();

    cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
    // base::synchronize();
  }

  template < typename body_type >
  void fused(IdxT len_outer, IdxT len_inner, IdxT len_hint, body_type&& body_in)
  {
    using decayed_body_type = typename std::decay<body_type>::type;

    constexpr IdxT threads0 = 1;
    constexpr IdxT threads1 = 1;
    constexpr IdxT threads2 = 1024;
    const IdxT blocks0 = len_outer;
    const IdxT blocks1 = len_inner;
    const IdxT blocks2 = (len_hint + threads2 - 1) / threads2;

#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
    group_type g = get_active_group();
    unsigned long long* kernel_starts = nullptr;
    unsigned long long* kernel_stops = nullptr;
    int kernel_id = -1;
    if (g.f) {
      if (g.f->m_do_timing) {
        assert(g.f->m_kernel_starts != nullptr);
        assert(g.f->m_kernel_id + len_outer*len_inner <= g.f->m_max_num_nodes);
        kernel_starts = g.f->m_kernel_starts;
        kernel_stops = g.f->m_kernel_starts + g.f->m_num_kernel_timers;
        kernel_id = g.f->m_kernel_id; g.f->m_kernel_id += len_outer*len_inner;
      }
      g.f->m_used = true;
      g.f->m_num_nodes += len_outer*len_inner;
      if (g.f->m_num_nodes > g.f->m_max_num_nodes) { g.f->m_max_num_nodes = g.f->m_num_nodes; }
    }
#endif

    void* func =(void*)&cuda_fused<decayed_body_type, threads0*threads1*threads2>;
    dim3 gridDim(blocks2, blocks1, blocks0);
    dim3 blockDim(threads2, threads1, threads0);
    void* args[]{&body_in
#ifdef COMB_CUDA_FUSED_KERNEL_DEVICE_TIMER
      , &kernel_starts, &kernel_stops, &kernel_id
#endif
      };
    size_t sharedMem = 0;
    cudaStream_t stream = base::stream_launch();
#ifdef COMB_CUDA_FUSED_KERNEL_HOST_TIMER
    std::chrono::time_point<std::chrono::high_resolution_clock> t0;
    if (g.f && g.f->m_do_timing) {
      if (kernel_id == 0) {
        cuda_spin_kernel<0><<<1,1,0,stream>>>(0.001); // spin for 0.001 seconds
        cuda_timer_kernel<0><<<1,1,0,stream>>>(g.f->m_kernel_pre_start);
        cudaCheck(cudaEventRecord(g.f->m_time_launch_events[0], stream));
      }
      t0 = std::chrono::high_resolution_clock::now();
    }
#endif
    cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
#ifdef COMB_CUDA_FUSED_KERNEL_HOST_TIMER
    if (g.f && g.f->m_do_timing) {
      auto t1 = std::chrono::high_resolution_clock::now();
      if (g.f->m_kernel_id >= g.f->m_max_num_nodes) {
        cudaCheck(cudaEventRecord(g.f->m_time_launch_events[1], stream));
        cuda_timer_kernel<0><<<1,1,0,stream>>>(g.f->m_kernel_pre_start+1);
      }
      g.f->m_time_launch += std::chrono::duration<double>(t1-t0).count();
    }
#endif
    // base::synchronize();
  }
};

#endif // COMB_ENABLE_CUDA

#endif // _POL_CUDA_HPP
