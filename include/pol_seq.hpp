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

#include "utils.hpp"

struct seq_pol {
  static const bool async = false;
  static const char* get_name() { return "seq"; }
  using event_type = int;
};

template < >
struct ExecContext<seq_pol>
{

};

inline bool operator==(ExecContext<seq_pol> const& lhs, ExecContext<seq_pol> const& rhs)
{
  COMB::ignore_unused(lhs, rhs);
  return true;
}

// synchronization functions
inline void synchronize(ExecContext<seq_pol> const&)
{
}

// force start functions
inline void persistent_launch(ExecContext<seq_pol> const&)
{
}

// force complete functions
inline void batch_launch(ExecContext<seq_pol> const&)
{
}

// force complete functions
inline void persistent_stop(ExecContext<seq_pol> const&)
{
}

// event creation functions
inline typename seq_pol::event_type createEvent(ExecContext<seq_pol> const&)
{
  return typename seq_pol::event_type{};
}

// event record functions
inline void recordEvent(ExecContext<seq_pol> const&, typename seq_pol::event_type)
{
}

// event query functions
inline bool queryEvent(ExecContext<seq_pol> const&, typename seq_pol::event_type)
{
  return true;
}

// event wait functions
inline void waitEvent(ExecContext<seq_pol> const&, typename seq_pol::event_type)
{
}

// event destroy functions
inline void destroyEvent(ExecContext<seq_pol> const&, typename seq_pol::event_type)
{
}

// for_all functions
template < typename body_type >
inline void for_all(ExecContext<seq_pol> const& con, IdxT begin, IdxT end, body_type&& body)
{
  COMB::ignore_unused(con);
  IdxT i = 0;
  for(IdxT i0 = begin; i0 < end; ++i0) {
    body(i0, i++);
  }
  //synchronize(con);
}

template < typename body_type >
void for_all_2d(ExecContext<seq_pol> const& con, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  COMB::ignore_unused(con);
  IdxT i = 0;
  for(IdxT i0 = begin0; i0 < end0; ++i0) {
    for(IdxT i1 = begin1; i1 < end1; ++i1) {
      body(i0, i1, i++);
    }
  }
  //synchronize(con);
}

template < typename body_type >
inline void for_all_3d(ExecContext<seq_pol> const& con, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  COMB::ignore_unused(con);
  IdxT i = 0;
  for(IdxT i0 = begin0; i0 < end0; ++i0) {
    for(IdxT i1 = begin1; i1 < end1; ++i1) {
      for(IdxT i2 = begin2; i2 < end2; ++i2) {
        body(i0, i1, i2, i++);
      }
    }
  }
  //synchronize(con);
}

#endif // _POL_SEQ_HPP
