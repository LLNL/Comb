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

#ifndef _POL_SEQ_HPP
#define _POL_SEQ_HPP

#include "config.hpp"

#include "utils.hpp"

struct seq_component
{
  void* ptr = nullptr;
}

struct seq_pol {
  static const bool async = false;
  static const char* get_name() { return "seq"; }
  using event_type = int;
  using component_type = seq_component;
};

template < >
struct ExecContext<seq_pol> : CPUContext
{
  using pol = seq_pol;
  using event_type = typename pol::event_type;
  using component_type = typename pol::component_type;

  using base = CPUContext;

  ExecContext()
    : base()
  { }

  ExecContext(base const& b)
    : base(b)
  { }

  // synchronization functions
  void synchronize()
  {
  }

  // force start functions
  void persistent_launch()
  {
  }

  // force complete functions
  void batch_launch()
  {
  }

  // force complete functions
  void persistent_stop()
  {
  }

  // event creation functions
  event_type createEvent()
  {
    return event_type{};
  }

  // event record functions
  void recordEvent(event_type)
  {
  }

  // event query functions
  bool queryEvent(event_type)
  {
    return true;
  }

  // event wait functions
  void waitEvent(event_type)
  {
  }

  // event destroy functions
  void destroyEvent(event_type)
  {
  }

  // for_all functions
  template < typename body_type >
  void for_all(IdxT begin, IdxT end, body_type&& body)
  {
    IdxT i = 0;
    for(IdxT i0 = begin; i0 < end; ++i0) {
      body(i0, i++);
    }
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
  {
    IdxT i = 0;
    for(IdxT i0 = begin0; i0 < end0; ++i0) {
      for(IdxT i1 = begin1; i1 < end1; ++i1) {
        body(i0, i1, i++);
      }
    }
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
  {
    IdxT i = 0;
    for(IdxT i0 = begin0; i0 < end0; ++i0) {
      for(IdxT i1 = begin1; i1 < end1; ++i1) {
        for(IdxT i2 = begin2; i2 < end2; ++i2) {
          body(i0, i1, i2, i++);
        }
      }
    }
    // base::synchronize();
  }

};

#endif // _POL_SEQ_HPP
