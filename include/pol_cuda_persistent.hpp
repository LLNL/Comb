//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

#ifdef COMB_ENABLE_CUDA
#include "persistent_launch.hpp"

struct cuda_persistent_pol {
  static const bool async = true;
  static const char* get_name() { return ( get_batch_always_grid_sync() ? "cudaPersistent" : "cudaPersistent_fewgs" ); }
  using event_type = detail::batch_event_type_ptr;
};

template < >
struct ExecContext<cuda_persistent_pol>
{
  cudaStream_t stream = 0;
};

inline bool operator==(ExecContext<cuda_persistent_pol> const& lhs, ExecContext<cuda_persistent_pol> const& rhs)
{
  return lhs.stream == rhs.stream;
}

inline void synchronize(ExecContext<cuda_persistent_pol> const& con)
{
  cuda::persistent_launch::synchronize(con.stream);
}

inline void persistent_launch(ExecContext<cuda_persistent_pol> const& con)
{
  cuda::persistent_launch::force_launch(con.stream);
}

inline void batch_launch(ExecContext<cuda_persistent_pol> const&)
{
}

inline void persistent_stop(ExecContext<cuda_persistent_pol> const& con)
{
  cuda::persistent_launch::force_stop(con.stream);
}

inline typename cuda_persistent_pol::event_type createEvent(ExecContext<cuda_persistent_pol> const&)
{
  return cuda::persistent_launch::createEvent();
}

inline void recordEvent(ExecContext<cuda_persistent_pol> const& con, typename cuda_persistent_pol::event_type event)
{
  return cuda::persistent_launch::recordEvent(event, con.stream);
}

inline bool queryEvent(ExecContext<cuda_persistent_pol> const&, typename cuda_persistent_pol::event_type event)
{
  return cuda::persistent_launch::queryEvent(event);
}

inline void waitEvent(ExecContext<cuda_persistent_pol> const&, typename cuda_persistent_pol::event_type event)
{
  cuda::persistent_launch::waitEvent(event);
}

inline void destroyEvent(ExecContext<cuda_persistent_pol> const&, typename cuda_persistent_pol::event_type event)
{
  cuda::persistent_launch::destroyEvent(event);
}

template < typename body_type >
inline void for_all(ExecContext<cuda_persistent_pol> const& con, IdxT begin, IdxT end, body_type&& body)
{
  COMB::ignore_unused(con);
  cuda::persistent_launch::for_all(begin, end, std::forward<body_type>(body), con.stream);
  //synchronize(con);
}

template < typename body_type >
inline void for_all_2d(ExecContext<cuda_persistent_pol> const& con, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  COMB::ignore_unused(con);
  IdxT len = (end0 - begin0) * (end1 - begin1);
  cuda::persistent_launch::for_all(0, len, detail::adapter_2d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, std::forward<body_type>(body)}, con.stream);
  //synchronize(con);
}

template < typename body_type >
inline void for_all_3d(ExecContext<cuda_persistent_pol> const& con, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  COMB::ignore_unused(con);
  IdxT len = (end0 - begin0) * (end1 - begin1) * (end2 - begin2);
  cuda::persistent_launch::for_all(0, len, detail::adapter_3d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body)}, con.stream);
  //synchronize(con);
}

#endif // COMB_ENABLE_CUDA

#endif // _POL_CUDA_PERSISTENT_HPP
