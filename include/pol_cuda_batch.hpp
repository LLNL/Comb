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

#ifndef _POL_CUDA_BATCH_HPP
#define _POL_CUDA_BATCH_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_CUDA
#include "batch_launch.hpp"

struct cuda_batch_pol {
  static const bool async = true;
  static const char* get_name() { return ( get_batch_always_grid_sync() ? "cudaBatch"      : "cudaBatch_fewgs"      ); }
  using event_type = detail::batch_event_type_ptr;
  using cache_type = int;
};

template < >
struct ExecContext<cuda_batch_pol> : CudaContext
{
  using pol = cuda_batch_pol;
  using event_type = typename pol::event_type;

  using base = CudaContext;

  ExecContext()
    : base()
  { }

  ExecContext(base const& b)
    : base(b)
  { }

  void synchronize()
  {
    cuda::batch_launch::synchronize(base::stream());
  }

  void persistent_launch()
  {
  }

  void batch_launch()
  {
    cuda::batch_launch::force_launch(base::stream());
  }

  void persistent_stop()
  {
  }

  event_type createEvent()
  {
    return cuda::batch_launch::createEvent();
  }

  void recordEvent(event_type event)
  {
    return cuda::batch_launch::recordEvent(event, base::stream());
  }

  bool queryEvent(event_type event)
  {
    return cuda::batch_launch::queryEvent(event);
  }

  void waitEvent(event_type event)
  {
    cuda::batch_launch::waitEvent(event);
  }

  void destroyEvent(event_type event)
  {
    cuda::batch_launch::destroyEvent(event);
  }

  template < typename body_type >
  void for_all(IdxT begin, IdxT end, body_type&& body)
  {
    cuda::batch_launch::for_all(begin, end, std::forward<body_type>(body), base::stream());
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
  {
    IdxT len = (end0 - begin0) * (end1 - begin1);
    cuda::batch_launch::for_all(0, len, detail::adapter_2d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, std::forward<body_type>(body)}, base::stream());
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
  {
    IdxT len = (end0 - begin0) * (end1 - begin1) * (end2 - begin2);
    cuda::batch_launch::for_all(0, len, detail::adapter_3d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body)}, base::stream());
    // base::synchronize();
  }

};

#endif // COMB_ENABLE_CUDA

#endif // _POL_CUDA_BATCH_HPP
