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

#ifndef _POL_OMP_HPP
#define _POL_OMP_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_OPENMP

#include <omp.h>

// #define COMB_USE_OMP_COLLAPSE
// #define COMB_USE_OMP_WEAK_COLLAPSE

#include "utils.hpp"
#include "memory.hpp"

struct omp_component
{
  void* ptr = nullptr;
};

struct omp_group
{
  void* ptr = nullptr;
};

struct omp_pol {
  static const bool async = false;
  static const char* get_name() { return "omp"; }
  using event_type = int;
  using component_type = omp_component;
  using group_type = omp_group;
};

template < >
struct ExecContext<omp_pol> : CPUContext
{
  using pol = omp_pol;
  using event_type = typename pol::event_type;
  using component_type = typename pol::component_type;
  using group_type = typename pol::group_type;

  using base = CPUContext;

  ExecContext()
    : base()
  { }

  ExecContext(base const& b)
    : base(b)
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

  void start_component(component_type)
  {

  }

  void finish_component(component_type)
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
    const IdxT len = end - begin;
  #pragma omp parallel for
    for(IdxT i = 0; i < len; ++i) {
      body(i + begin, i);
    }
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
  {
    const IdxT len0 = end0 - begin0;
    const IdxT len1 = end1 - begin1;

  #ifdef COMB_USE_OMP_COLLAPSE

  #pragma omp parallel for collapse(2)
    for(IdxT i0 = 0; i0 < len0; ++i0) {
      for(IdxT i1 = 0; i1 < len1; ++i1) {
        IdxT i = i0 * len1 + i1;
        body(i0 + begin0, i1 + begin1, i);
      }
    }

  #elif defined(COMB_USE_OMP_WEAK_COLLAPSE)

    const IdxT len01 = len0 * len1;

  #pragma omp parallel
    {
      IdxT nthreads = omp_get_num_threads();
      IdxT threadid = omp_get_thread_num();


      const IdxT stride = (len01 + nthreads - 1) / nthreads;

      IdxT i = threadid * stride;
      IdxT iend = (threadid + 1) * stride;
      if (iend > len01) iend = len01;


      const IdxT tmp1 = i / len1;
      IdxT i1 = i - tmp1 * len1;

      IdxT i0 = tmp1;

      for (; i < iend; ++i) {

        body(i0 + begin0, i1 + begin1, i);

        i1 += 1;

        if (i1 >= len1) {
          i1 -= len1;

          i0 += 1;
        }
      }
    }

  #else

  #pragma omp parallel for
    for(IdxT i0 = 0; i0 < len0; ++i0) {
      for(IdxT i1 = 0; i1 < len1; ++i1) {
        IdxT i = i0 * len1 + i1;
        body(i0 + begin0, i1 + begin1, i);
      }
    }

  #endif
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
  {
    const IdxT len0 = end0 - begin0;
    const IdxT len1 = end1 - begin1;
    const IdxT len2 = end2 - begin2;
    const IdxT len12 = len1 * len2;

  #ifdef COMB_USE_OMP_COLLAPSE

  #pragma omp parallel for collapse(3)
    for(IdxT i0 = 0; i0 < len0; ++i0) {
      for(IdxT i1 = 0; i1 < len1; ++i1) {
        for(IdxT i2 = 0; i2 < len2; ++i2) {
          IdxT i = i0 * len12 + i1 * len2 + i2;
          body(i0 + begin0, i1 + begin1, i2 + begin2, i);
        }
      }
    }

  #elif defined(COMB_USE_OMP_WEAK_COLLAPSE)

    const IdxT len012 = len0 * len1 * len2;

  #pragma omp parallel
    {
      IdxT nthreads = omp_get_num_threads();
      IdxT threadid = omp_get_thread_num();


      const IdxT stride = (len012 + nthreads - 1) / nthreads;

      IdxT i = threadid * stride;
      IdxT iend = (threadid + 1) * stride;
      if (iend > len012) iend = len012;


      const IdxT tmp2 = i / len2;
      IdxT i2 = i - tmp2 * len2;

      const IdxT tmp1 = tmp2 / len1;
      IdxT i1 = tmp2 - tmp1 * len1;

      IdxT i0 = tmp1;

      for (; i < iend; ++i) {

        body(i0 + begin0, i1 + begin1, i2 + begin2, i);

        i2 += 1;

        if (i2 >= len2) {
          i2 -= len2;

          i1 += 1;

          if (i1 >= len1) {
            i1 -= len1;

            i0 += 1;
          }
        }
      }
    }

  #else

  #pragma omp parallel for
    for(IdxT i0 = 0; i0 < len0; ++i0) {
      for(IdxT i1 = 0; i1 < len1; ++i1) {
        for(IdxT i2 = 0; i2 < len2; ++i2) {
          IdxT i = i0 * len12 + i1 * len2 + i2;
          body(i0 + begin0, i1 + begin1, i2 + begin2, i);
        }
      }
    }

  #endif
    // base::synchronize();
  }

};

#endif // COMB_ENABLE_OPENMP

#endif // _POL_OMP_HPP
