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

#ifdef COMB_ENABLE_CUDA_GRAPH
#include "graph_launch.hpp"

struct cuda_graph_pol {
  static const bool async = true;
  static const char* get_name() { return "cudaGraph"; }
  using event_type = cuda::graph_launch::event_type;
};

template < >
struct ExecContext<cuda_graph_pol>
{
  cudaStream_t stream = 0;
};

inline bool operator==(ExecContext<cuda_graph_pol> const& lhs, ExecContext<cuda_graph_pol> const& rhs)
{
  return lhs.stream == rhs.stream;
}

inline void synchronize(ExecContext<cuda_graph_pol> const& con)
{
  cuda::graph_launch::synchronize(con.stream);
}

inline void persistent_launch(ExecContext<cuda_graph_pol> const&)
{
}

inline void batch_launch(ExecContext<cuda_graph_pol> const& con)
{
  cuda::graph_launch::force_launch(con.stream);
}

inline void persistent_stop(ExecContext<cuda_graph_pol> const&)
{
}

inline typename cuda_graph_pol::event_type createEvent(ExecContext<cuda_graph_pol> const&)
{
  return cuda::graph_launch::createEvent();
}

inline void recordEvent(ExecContext<cuda_graph_pol> const& con, typename cuda_graph_pol::event_type event)
{
  return cuda::graph_launch::recordEvent(event, con.stream);
}

inline bool queryEvent(ExecContext<cuda_graph_pol> const&, typename cuda_graph_pol::event_type event)
{
  return cuda::graph_launch::queryEvent(event);
}

inline void waitEvent(ExecContext<cuda_graph_pol> const&, typename cuda_graph_pol::event_type event)
{
  cuda::graph_launch::waitEvent(event);
}

inline void destroyEvent(ExecContext<cuda_graph_pol> const&, typename cuda_graph_pol::event_type event)
{
  cuda::graph_launch::destroyEvent(event);
}

template < typename body_type >
inline void for_all(ExecContext<cuda_graph_pol> const& con, IdxT begin, IdxT end, body_type&& body)
{
  COMB::ignore_unused(con);
  cuda::graph_launch::for_all(begin, end, std::forward<body_type>(body));
  //synchronize(con);
}

template < typename body_type >
inline void for_all_2d(ExecContext<cuda_graph_pol> const& con, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  COMB::ignore_unused(con);
  cuda::graph_launch::for_all_2d(begin0, end0, begin1, end1, std::forward<body_type>(body));
  //synchronize(con);
}

template < typename body_type >
inline void for_all_3d(ExecContext<cuda_graph_pol> const& con, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  COMB::ignore_unused(con);
  cuda::graph_launch::for_all_3d(begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body));
  //synchronize(con);
}

#endif // COMB_ENABLE_CUDA_GRAPH

#endif // _POL_CUDA_GRAPH_HPP
