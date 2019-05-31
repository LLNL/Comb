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

#ifndef _POL_CUDA_BATCH_HPP
#define _POL_CUDA_BATCH_HPP

#ifdef COMB_ENABLE_CUDA
#include "batch_launch.hpp"

struct cuda_batch_pol {
  static const bool async = true;
  static const char* get_name() { return ( get_batch_always_grid_sync() ? "cudaBatch"      : "cudaBatch_fewgs"      ); }
  using event_type = detail::batch_event_type_ptr;
};

inline void synchronize(cuda_batch_pol const&)
{
  cuda::batch_launch::synchronize();
}

inline void persistent_launch(cuda_batch_pol const&)
{
}

inline void batch_launch(cuda_batch_pol const&)
{
  cuda::batch_launch::force_launch();
}

inline void persistent_stop(cuda_batch_pol const&)
{
}

inline typename cuda_batch_pol::event_type createEvent(cuda_batch_pol const&)
{
  return cuda::batch_launch::createEvent();
}

inline void recordEvent(cuda_batch_pol const&, typename cuda_batch_pol::event_type event)
{
  return cuda::batch_launch::recordEvent(event);
}

inline bool queryEvent(cuda_batch_pol const&, typename cuda_batch_pol::event_type event)
{
  return cuda::batch_launch::queryEvent(event);
}

inline void waitEvent(cuda_batch_pol const&, typename cuda_batch_pol::event_type event)
{
  cuda::batch_launch::waitEvent(event);
}

inline void destroyEvent(cuda_batch_pol const&, typename cuda_batch_pol::event_type event)
{
  cuda::batch_launch::destroyEvent(event);
}

template < typename body_type >
inline void for_all(cuda_batch_pol const& pol, IdxT begin, IdxT end, body_type&& body)
{
  COMB::ignore_unused(pol);
  cuda::batch_launch::for_all(begin, end, std::forward<body_type>(body));
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_2d(cuda_batch_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  COMB::ignore_unused(pol);
  IdxT len = (end0 - begin0) * (end1 - begin1);
  cuda::batch_launch::for_all(0, len, detail::adapter_2d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, std::forward<body_type>(body)});
  //synchronize(pol);
}

template < typename body_type >
inline void for_all_3d(cuda_batch_pol const& pol, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  COMB::ignore_unused(pol);
  IdxT len = (end0 - begin0) * (end1 - begin1) * (end2 - begin2);
  cuda::batch_launch::for_all(0, len, detail::adapter_3d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body)});
  //synchronize(pol);
}

#endif // COMB_ENABLE_CUDA

#endif // _POL_CUDA_BATCH_HPP
