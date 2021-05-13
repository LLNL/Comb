//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
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

#ifndef _POL_RAJA_HPP
#define _POL_RAJA_HPP

#include "config.hpp"

#include "memory.hpp"

#ifdef COMB_ENABLE_RAJA
#include "RAJA/RAJA.hpp"

namespace raja
{

struct base { };

struct seq : base
{
  static const bool async = false;
  static const char* get_name() { return "raja_seq"; }
  using resource_type = RAJA::resources::Host;
  using for_all_policy = RAJA::loop_exec;
  using for_all_2d_policy =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::For<1, RAJA::loop_exec,
            RAJA::statement::Lambda<0> > > >;
  using for_all_3d_policy =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::For<1, RAJA::loop_exec,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<0> > > > >;
  using fused_policy =
      RAJA::WorkGroupPolicy<
          RAJA::loop_work,
          RAJA::ordered,
          RAJA::ragged_array_of_objects>;
};

#ifdef COMB_ENABLE_OPENMP

struct omp : base
{
  static const bool async = false;
  static const char* get_name() { return "raja_omp"; }
  using resource_type = RAJA::resources::Host;
  using for_all_policy = RAJA::loop_exec;
#if defined(COMB_USE_OMP_COLLAPSE) || defined(COMB_USE_OMP_WEAK_COLLAPSE)
  using for_all_2d_policy =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0> > >;
  using for_all_3d_policy =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1, 2>,
          RAJA::statement::Lambda<0> > >;
#else
  using for_all_2d_policy =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
          RAJA::statement::For<1, RAJA::loop_exec,
            RAJA::statement::Lambda<0> > > >;
  using for_all_3d_policy =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
          RAJA::statement::For<1, RAJA::loop_exec,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<0> > > > >;
#endif
  using fused_policy =
      RAJA::WorkGroupPolicy<
          RAJA::omp_work,
          RAJA::ordered,
          RAJA::ragged_array_of_objects>;
};

#endif // COMB_ENABLE_OPENMP

#ifdef COMB_ENABLE_CUDA

struct cuda : base
{
  static const bool async = true;
  static const char* get_name() { return "raja_cuda"; }
  using resource_type = RAJA::resources::Cuda;
  using for_all_policy = RAJA::cuda_exec_async<256>;
  using for_all_2d_policy =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<256,
          RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::cuda_block_x_direct,
            RAJA::statement::Tile<1, RAJA::tile_fixed<8>, RAJA::cuda_block_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::Lambda<0> > > > > > >;
  using for_all_3d_policy =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<256,
          RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::cuda_block_x_direct,
            RAJA::statement::Tile<1, RAJA::tile_fixed<8>, RAJA::cuda_block_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<2, RAJA::cuda_block_z_direct,
                    RAJA::statement::Lambda<0> > > > > > > >;
  using fused_policy =
      RAJA::WorkGroupPolicy<
          RAJA::cuda_work_async<1024>,
          RAJA::unordered_cuda_loop_y_block_iter_x_threadblock_average,
          RAJA::constant_stride_array_of_objects>;
};

#endif // COMB_ENABLE_CUDA

}

struct raja_component
{
  void* ptr = nullptr;
};

struct raja_group
{
  void* ptr = nullptr;
};

template < typename base_policy >
struct raja_pol : base_policy {
  // using event_type = decltype(camp::val<resource_type>().get_event());
  using event_type = RAJA::resources::Event;
  using component_type = raja_component;
  using group_type = raja_group;
};

using raja_seq_pol  = raja_pol<raja::seq>;
#ifdef COMB_ENABLE_OPENMP
using raja_omp_pol  = raja_pol<raja::omp>;
#endif
#ifdef COMB_ENABLE_CUDA
using raja_cuda_pol = raja_pol<raja::cuda>;
#endif


template < typename base_policy >
struct ExecContext<raja_pol<base_policy>> : RAJAContext<typename base_policy::resource_type>
{
  using pol = raja_pol<base_policy>;
  using event_type = typename pol::event_type;
  using component_type = typename pol::component_type;
  using group_type = typename pol::group_type;

  using resource_type = typename pol::resource_type;
  using for_all_policy = typename pol::for_all_policy;
  using for_all_2d_policy = typename pol::for_all_2d_policy;
  using for_all_3d_policy = typename pol::for_all_3d_policy;
  using fused_policy = typename pol::fused_policy;
  using fused_allocator = COMB::std_allocator<char>;

  using base = RAJAContext<resource_type>;

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

  void synchronize()
  {
    base::synchronize();
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

  event_type createEvent()
  {
    return event_type{};
  }

  void recordEvent(event_type& event)
  {
    event = base::resource().get_event();
  }

  void finish_component_recordEvent(group_type group, component_type component, event_type& event)
  {
    finish_component(group, component);
    recordEvent(event);
  }

  bool queryEvent(event_type& event)
  {
    return event.check();
  }

  void waitEvent(event_type& event)
  {
    event.wait();
  }

  void destroyEvent(event_type& event)
  {
    event = event_type{};
  }

  template < typename body_type >
  void for_all(IdxT len, body_type&& body)
  {
    RAJA::TypedRangeSegment<IdxT> seg(0, len);

    RAJA::forall<for_all_policy>(base::res_launch(), seg, body);
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT len0, IdxT len1, body_type&& body)
  {
    RAJA::TypedRangeSegment<IdxT> seg0(0, len0);
    RAJA::TypedRangeSegment<IdxT> seg1(0, len1);

    RAJA::kernel_resource<for_all_2d_policy>(
        RAJA::make_tuple(seg0, seg1), base::res_launch(), body);
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT len0, IdxT len1, IdxT len2, body_type&& body)
  {
    RAJA::TypedRangeSegment<IdxT> seg0(0, len0);
    RAJA::TypedRangeSegment<IdxT> seg1(0, len1);
    RAJA::TypedRangeSegment<IdxT> seg2(0, len2);

    RAJA::kernel_resource<for_all_3d_policy>(
        RAJA::make_tuple(seg0, seg1, seg2), base::res_launch(), body);
    // base::synchronize();
  }

};

#endif // COMB_ENABLE_CUDA

#endif // _POL_RAJA_HPP
