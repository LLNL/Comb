//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2022, Lawrence Livermore National Security, LLC.
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

#ifndef _POL_SEQ_HPP
#define _POL_SEQ_HPP

#include "config.hpp"

#include "exec_utils.hpp"
#include "memory.hpp"

struct seq_component
{
  void* ptr = nullptr;
};

struct seq_group
{
  void* ptr = nullptr;
};

struct seq_pol {
  static const bool async = false;
  static const char* get_name() { return "seq"; }
  using event_type = int;
  using component_type = seq_component;
  using group_type = seq_group;
};

template < >
struct ExecContext<seq_pol> : CPUContext
{
  using pol = seq_pol;
  using event_type = typename pol::event_type;
  using component_type = typename pol::component_type;
  using group_type = typename pol::group_type;

  using base = CPUContext;

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

  // synchronization functions
  void synchronize()
  {
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

  // event creation functions
  event_type createEvent()
  {
    return event_type{};
  }

  // event record functions
  void recordEvent(event_type&)
  {
  }

  void finish_component_recordEvent(group_type group, component_type component, event_type& event)
  {
    finish_component(group, component);
    recordEvent(event);
  }

  // event query functions
  bool queryEvent(event_type&)
  {
    return true;
  }

  // event wait functions
  void waitEvent(event_type&)
  {
  }

  // event destroy functions
  void destroyEvent(event_type&)
  {
  }

  // for_all functions
  template < typename body_type >
  void for_all(IdxT len, body_type&& body)
  {
    for(IdxT i0 = 0; i0 < len; ++i0) {
      body(i0);
    }
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT len0, IdxT len1, body_type&& body)
  {
    for(IdxT i0 = 0; i0 < len0; ++i0) {
      for(IdxT i1 = 0; i1 < len1; ++i1) {
        body(i0, i1);
      }
    }
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT len0, IdxT len1, IdxT len2, body_type&& body)
  {
    for(IdxT i0 = 0; i0 < len0; ++i0) {
      for(IdxT i1 = 0; i1 < len1; ++i1) {
        for(IdxT i2 = 0; i2 < len2; ++i2) {
          body(i0, i1, i2);
        }
      }
    }
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
        for (IdxT i = 0; i < body.len; ++i) {
          body(i);
        }
      }
    }
    // base::synchronize();
  }


};

#endif // _POL_SEQ_HPP
