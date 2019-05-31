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

inline void synchronize(cuda_persistent_pol const&)
{
  cuda::persistent_launch::synchronize();
}

inline void persistent_launch(cuda_persistent_pol const&)
{
  cuda::persistent_launch::force_launch();
}

inline void batch_launch(cuda_persistent_pol const&)
{
}

inline void persistent_stop(cuda_persistent_pol const&)
{
  cuda::persistent_launch::force_stop();
}

inline typename cuda_persistent_pol::event_type createEvent(cuda_persistent_pol const&)
{
  return cuda::persistent_launch::createEvent();
}

inline void recordEvent(cuda_persistent_pol const&, typename cuda_persistent_pol::event_type event)
{
  return cuda::persistent_launch::recordEvent(event);
}

inline bool queryEvent(cuda_persistent_pol const&, typename cuda_persistent_pol::event_type event)
{
  return cuda::persistent_launch::queryEvent(event);
}

inline void waitEvent(cuda_persistent_pol const&, typename cuda_persistent_pol::event_type event)
{
  cuda::persistent_launch::waitEvent(event);
}

inline void destroyEvent(cuda_persistent_pol const&, typename cuda_persistent_pol::event_type event)
{
  cuda::persistent_launch::destroyEvent(event);
}

template < typename body_type >
inline void for_all(cuda_persistent_pol const& pol, IdxT begin, IdxT end, body_type&& body)
{
  COMB::ignore_unused(pol);
  cuda::persistent_launch::for_all(begin, end, std::forward<body_type>(body));
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_2d(cuda_persistent_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  COMB::ignore_unused(pol);
  IdxT len = (end0 - begin0) * (end1 - begin1);
  cuda::persistent_launch::for_all(0, len, detail::adapter_2d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, std::forward<body_type>(body)});
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_3d(cuda_persistent_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  COMB::ignore_unused(pol);
  IdxT len = (end0 - begin0) * (end1 - begin1) * (end2 - begin2);
  cuda::persistent_launch::for_all(0, len, detail::adapter_3d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body)});
  //synchronize(pol);
}

#endif // COMB_ENABLE_CUDA

#endif // _POL_CUDA_PERSISTENT_HPP
