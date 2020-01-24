//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

#ifndef _POL_CUDA_GRAPH_HPP
#define _POL_CUDA_GRAPH_HPP

#include "config.hpp"

#include "memory.hpp"

#ifdef COMB_ENABLE_CUDA_GRAPH
#include "graph_launch.hpp"

struct cuda_graph_pol {
  static const bool async = true;
  static const char* get_name() { return "cudaGraph"; }
  using event_type = cuda::graph_launch::event_type;
  using component_type = cuda::graph_launch::component;
  using group_type = cuda::graph_launch::group;
};

template < >
struct ExecContext<cuda_graph_pol> : CudaContext
{
  using pol = cuda_graph_pol;
  using event_type = typename pol::event_type;
  using component_type = typename pol::component_type;
  using group_type = typename pol::group_type;

  using base = CudaContext;

  COMB::Allocator& util_aloc;

#ifdef COMB_GRAPH_KERNEL_LAUNCH_COMPONENT_STREAMS
  component_type m_component;
#endif


  ExecContext(base const& b, COMB::Allocator& util_aloc_)
    : base(b)
    , util_aloc(util_aloc_)
#ifdef COMB_GRAPH_KERNEL_LAUNCH_COMPONENT_STREAMS
    , m_component{base(*this)}
#endif
  { }

  void ensure_waitable()
  {
    cuda::graph_launch::force_launch(base::stream_launch());
  }

  template < typename context >
  void waitOn(context& con)
  {
    con.ensure_waitable();
    base::waitOn(con);
  }

  void synchronize()
  {
    cuda::graph_launch::synchronize(base::stream_launch());
  }

  group_type create_group()
  {
    return cuda::graph_launch::create_group();
  }

  void start_group(group_type group)
  {
    cuda::graph_launch::set_active_group(group);
  }

  void finish_group(group_type)
  {
    cuda::graph_launch::force_launch(base::stream_launch());
  }

  void destroy_group(group_type group)
  {
    cuda::graph_launch::destroy_group(group);
  }

  component_type create_component()
  {
    return component_type{};
  }

  void start_component(group_type, component_type component)
  {
#ifdef COMB_GRAPH_KERNEL_LAUNCH_COMPONENT_STREAMS
    m_component = component;
    m_component.m_con.waitOn(base(*this));
#endif
  }

  void finish_component(group_type, component_type component)
  {
#ifdef COMB_GRAPH_KERNEL_LAUNCH_COMPONENT_STREAMS
    base::waitOn(component.m_con);
    m_component.m_con = base(*this);
#endif
  }

  void destroy_component(component_type)
  {
#ifdef COMB_GRAPH_KERNEL_LAUNCH_COMPONENT_STREAMS
    m_component.m_con = base(*this);
#endif
  }

  event_type createEvent()
  {
    return cuda::graph_launch::createEvent();
  }

  void recordEvent(event_type event)
  {
    return cuda::graph_launch::recordEvent(event, base::stream());
  }

  void finish_component_recordEvent(group_type group, component_type component, event_type event)
  {
    finish_component(group, component);
    recordEvent(event);
  }

  bool queryEvent(event_type event)
  {
    return cuda::graph_launch::queryEvent(event);
  }

  void waitEvent(event_type event)
  {
    cuda::graph_launch::waitEvent(event);
  }

  void destroyEvent(event_type event)
  {
    cuda::graph_launch::destroyEvent(event);
  }

  template < typename body_type >
  void for_all(IdxT begin, IdxT end, body_type&& body)
  {
    cuda::graph_launch::for_all(begin, end, std::forward<body_type>(body)
#ifdef COMB_GRAPH_KERNEL_LAUNCH
        , m_component.m_con.stream_launch()
#endif
        );
    // m_component.m_con.synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
  {
    cuda::graph_launch::for_all_2d(begin0, end0, begin1, end1, std::forward<body_type>(body)
#ifdef COMB_GRAPH_KERNEL_LAUNCH
        , m_component.m_con.stream_launch()
#endif
        );
    // m_component.m_con.synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
  {
    cuda::graph_launch::for_all_3d(begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body)
#ifdef COMB_GRAPH_KERNEL_LAUNCH
        , m_component.m_con.stream_launch()
#endif
        );
    // m_component.m_con.synchronize();
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
        cuda::graph_launch::for_all(0, body.len, body
#ifdef COMB_GRAPH_KERNEL_LAUNCH
            , m_component.m_con.stream_launch()
#endif
            );
      }
    }
    // m_component.m_con.synchronize();
  }

};

#endif // COMB_ENABLE_CUDA_GRAPH

#endif // _POL_CUDA_GRAPH_HPP
