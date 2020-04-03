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

#ifndef _POL_CUDA_PERSISTENT_HPP
#define _POL_CUDA_PERSISTENT_HPP

#include "config.hpp"

#include "memory.hpp"

#ifdef COMB_ENABLE_CUDA
#include "persistent_launch.hpp"

struct cuda_persistent_component
{
  void* ptr = nullptr;
};

struct cuda_persistent_group
{
  void* ptr = nullptr;
};

struct cuda_persistent_pol {
  static const bool async = true;
  static const char* get_name() { return ( get_batch_always_grid_sync() ? "cudaPersistent" : "cudaPersistent_fewgs" ); }
  using event_type = detail::batch_event_type_ptr;
  using component_type = cuda_persistent_component;
  using group_type = cuda_persistent_group;
};

template < >
struct ExecContext<cuda_persistent_pol> : CudaContext
{
  using pol = cuda_persistent_pol;
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
    cuda::persistent_launch::force_stop(base::stream_launch());
  }

  template < typename context >
  void waitOn(context& con)
  {
    con.ensure_waitable();
    base::waitOn(con);
  }

  void synchronize()
  {
    cuda::persistent_launch::synchronize(base::stream_launch());
  }

  group_type create_group()
  {
    return group_type{};
  }

  void start_group(group_type)
  {
    cuda::persistent_launch::force_launch(base::stream());
  }

  void finish_group(group_type)
  {
    cuda::persistent_launch::force_stop(base::stream_launch());
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
    return cuda::persistent_launch::createEvent();
  }

  void recordEvent(event_type event)
  {
    return cuda::persistent_launch::recordEvent(event, base::stream());
  }

  void finish_component_recordEvent(group_type group, component_type component, event_type event)
  {
    finish_component(group, component);
    recordEvent(event);
  }

  bool queryEvent(event_type event)
  {
    return cuda::persistent_launch::queryEvent(event);
  }

  void waitEvent(event_type event)
  {
    cuda::persistent_launch::waitEvent(event);
  }

  void destroyEvent(event_type event)
  {
    cuda::persistent_launch::destroyEvent(event);
  }

  template < typename body_type >
  void for_all(IdxT begin, IdxT end, body_type&& body)
  {
    cuda::persistent_launch::for_all(begin, end, std::forward<body_type>(body), base::stream());
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
  {
    IdxT len = (end0 - begin0) * (end1 - begin1);
    cuda::persistent_launch::for_all(0, len, detail::adapter_2d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, std::forward<body_type>(body)}, base::stream());
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
  {
    IdxT len = (end0 - begin0) * (end1 - begin1) * (end2 - begin2);
    cuda::persistent_launch::for_all(0, len, detail::adapter_3d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body)}, base::stream());
    // base::synchronize();
  }

  template < typename body_type >
  void fused(IdxT len_outer, IdxT len_inner, IdxT len_hint, body_type&& body_in)
  {
    COMB::ignore_unused(len_hint);
    for (IdxT i_outer = 0; i_outer < len_outer; ++i_outer) {
      auto body = body_in;
      body.set_outer(i_outer);
      for (IdxT i_inner = 0; i_inner < len_inner; ++i_inner) {
        body.set_inner(i_inner);
        cuda::persistent_launch::for_all(0, body.len, body, base::stream());
      }
    }
    // base::synchronize();
  }

};

#endif // COMB_ENABLE_CUDA

#endif // _POL_CUDA_PERSISTENT_HPP
