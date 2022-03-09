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

#ifndef _POL_HIP_HPP
#define _POL_HIP_HPP

#include "config.hpp"

#include "memory.hpp"

#ifdef COMB_ENABLE_HIP
#include "exec_utils_hip.hpp"

template < typename body_type >
__global__
void hip_for_all(IdxT len, body_type body)
{
  const IdxT i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < len) {
    body(i);
  }
}

template < typename body_type >
__global__
void hip_for_all_2d(IdxT len0, IdxT len1, body_type body)
{
  const IdxT i0 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      body(i0, i1);
    }
  }
}

template < typename body_type >
__global__
void hip_for_all_3d(IdxT len0, IdxT len1, IdxT len2, body_type body)
{
  const IdxT i0 = blockIdx.z;
  const IdxT i1 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i2 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      if (i2 < len2) {
        body(i0, i1, i2);
      }
    }
  }
}

template < typename body_type, IdxT num_threads >
__global__ __launch_bounds__(num_threads)
void hip_fused(body_type body_in)
{
  const IdxT i_outer = blockIdx.z;
  const IdxT i_inner = blockIdx.y;
  const IdxT ii      = threadIdx.x + blockIdx.x * blockDim.x;
  const IdxT i_stride = blockDim.x * gridDim.x;
  auto body = body_in;
  body.set_outer(i_outer);
  body.set_inner(i_inner);
  for (IdxT i = ii; i < body.len; i += i_stride) {
    body(i);
  }
}



struct hip_component
{
  void* ptr = nullptr;
};

struct hip_group
{
  void* ptr = nullptr;
};

struct hip_pol {
  static const bool async = true;
  static const char* get_name() { return "hip"; }
  using event_type = hipEvent_t;
  using component_type = hip_component;
  using group_type = hip_group;
};

template < >
struct ExecContext<hip_pol> : HipContext
{
  using pol = hip_pol;
  using event_type = typename pol::event_type;
  using component_type = typename pol::component_type;
  using group_type = typename pol::group_type;

  using base = HipContext;

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

  group_type create_group()
  {
    return group_type{};
  }

  void start_group(group_type)
  {
  }

  void finish_group(group_type)
  {
  }

  void destroy_group(group_type)
  {

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
    hipEvent_t event;
    hipCheck(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    return event;
  }

  void recordEvent(event_type& event)
  {
    hipCheck(hipEventRecord(event, base::stream()));
  }

  void finish_component_recordEvent(group_type group, component_type component, event_type& event)
  {
    finish_component(group, component);
    recordEvent(event);
  }

  bool queryEvent(event_type& event)
  {
    return hipCheckReady(hipEventQuery(event));
  }

  void waitEvent(event_type& event)
  {
    hipCheck(hipEventSynchronize(event));
  }

  void destroyEvent(event_type& event)
  {
    hipCheck(hipEventDestroy(event));
  }

  template < typename body_type >
  void for_all(IdxT len, body_type&& body)
  {
    using decayed_body_type = typename std::decay<body_type>::type;

    const IdxT threads = 256;
    const IdxT blocks = (len + threads - 1) / threads;

    void* func = (void*)&hip_for_all<decayed_body_type>;
    dim3 gridDim(blocks);
    dim3 blockDim(threads);
    void* args[]{&len, &body};
    size_t sharedMem = 0;
    hipStream_t stream = base::stream_launch();

    hipCheck(hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT len0, IdxT len1, body_type&& body)
  {
    using decayed_body_type = typename std::decay<body_type>::type;

    const IdxT threads0 = 8;
    const IdxT threads1 = 32;
    const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
    const IdxT blocks1 = (len1 + threads1 - 1) / threads1;

    void* func = (void*)&hip_for_all_2d<decayed_body_type>;
    dim3 gridDim(blocks1, blocks0, 1);
    dim3 blockDim(threads1, threads0, 1);
    void* args[]{&len0, &len1, &body};
    size_t sharedMem = 0;
    hipStream_t stream = base::stream_launch();

    hipCheck(hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT len0, IdxT len1, IdxT len2, body_type&& body)
  {
    using decayed_body_type = typename std::decay<body_type>::type;

    const IdxT threads0 = 1;
    const IdxT threads1 = 8;
    const IdxT threads2 = 32;
    const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
    const IdxT blocks1 = (len1 + threads1 - 1) / threads1;
    const IdxT blocks2 = (len2 + threads2 - 1) / threads2;

    void* func =(void*)&hip_for_all_3d<decayed_body_type>;
    dim3 gridDim(blocks2, blocks1, blocks0);
    dim3 blockDim(threads2, threads1, threads0);
    void* args[]{&len0, &len1, &len2, &body};
    size_t sharedMem = 0;
    hipStream_t stream = base::stream_launch();

    hipCheck(hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
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

    void* func =(void*)&hip_fused<decayed_body_type, threads0*threads1*threads2>;
    dim3 gridDim(blocks2, blocks1, blocks0);
    dim3 blockDim(threads2, threads1, threads0);
    void* args[]{&body_in};
    size_t sharedMem = 0;
    hipStream_t stream = base::stream_launch();

    hipCheck(hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
    // base::synchronize();
  }

};

#endif // COMB_ENABLE_HIP

#endif // _POL_HIP_HPP
