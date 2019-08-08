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

  ExecContext()
    : base()
  { }

  ExecContext(base const& b)
    : base(b)
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

  void start_component(component_type)
  {

  }

  void finish_component(component_type)
  {

  }

  void destroy_component(component_type)
  {

  }

  event_type createEvent()
  {
    return cuda::graph_launch::createEvent();
  }

  void recordEvent(event_type event)
  {
    return cuda::graph_launch::recordEvent(event, base::stream());
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
    cuda::graph_launch::for_all(begin, end, std::forward<body_type>(body));
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
  {
    cuda::graph_launch::for_all_2d(begin0, end0, begin1, end1, std::forward<body_type>(body));
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
  {
    cuda::graph_launch::for_all_3d(begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body));
    // base::synchronize();
  }

};

#endif // COMB_ENABLE_CUDA_GRAPH

#endif // _POL_CUDA_GRAPH_HPP
