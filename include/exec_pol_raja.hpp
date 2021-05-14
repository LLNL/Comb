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
          RAJA::statement::Tile<0, RAJA::tile_fixed<8>, RAJA::cuda_block_y_direct,
            RAJA::statement::Tile<1, RAJA::tile_fixed<32>, RAJA::cuda_block_x_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<0> > > > > > >;
  using for_all_3d_policy =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<256,
          RAJA::statement::For<0, RAJA::cuda_block_z_direct,
            RAJA::statement::Tile<1, RAJA::tile_fixed<8>, RAJA::cuda_block_y_direct,
              RAJA::statement::Tile<2, RAJA::tile_fixed<32>, RAJA::cuda_block_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<2, RAJA::cuda_thread_x_direct,
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
  {
    LOGPRINTF("%p ExecContext<raja>::ExecContext\n", this);
  }

  void ensure_waitable()
  {
    LOGPRINTF("%p ExecContext<raja>::ensure_waitable\n", this);
  }

  template < typename context >
  void waitOn(context& con)
  {
    LOGPRINTF("%p ExecContext<raja>::waitOn\n", this);
    con.ensure_waitable();
    base::waitOn(con);
  }

  void synchronize()
  {
    LOGPRINTF("%p ExecContext<raja>::synchronize\n", this);
    base::synchronize();
  }

  group_type create_group()
  {
    LOGPRINTF("%p ExecContext<raja>::create_group\n", this);
    return group_type{};
  }

  void start_group(group_type)
  {
    LOGPRINTF("%p ExecContext<raja>::start_group\n", this);
  }

  void finish_group(group_type)
  {
    LOGPRINTF("%p ExecContext<raja>::finish_group\n", this);
  }

  void destroy_group(group_type)
  {
    LOGPRINTF("%p ExecContext<raja>::destroy_group\n", this);
  }

  component_type create_component()
  {
    LOGPRINTF("%p ExecContext<raja>::create_component\n", this);
    return component_type{};
  }

  void start_component(group_type, component_type)
  {
    LOGPRINTF("%p ExecContext<raja>::start_component\n", this);
  }

  void finish_component(group_type, component_type)
  {
    LOGPRINTF("%p ExecContext<raja>::finish_component\n", this);
  }

  void destroy_component(component_type)
  {
    LOGPRINTF("%p ExecContext<raja>::destroy_component\n", this);
  }

  event_type createEvent()
  {
    LOGPRINTF("%p ExecContext<raja>::createEvent\n", this);
    return event_type{};
  }

  void recordEvent(event_type& event)
  {
    LOGPRINTF("%p ExecContext<raja>::recordEvent event %p\n", this, &event);
    event = base::resource().get_event_erased();
  }

  void finish_component_recordEvent(group_type group, component_type component, event_type& event)
  {
    LOGPRINTF("%p ExecContext<raja>::finish_component_recordEvent event %p\n", this, &event);
    finish_component(group, component);
    recordEvent(event);
  }

  bool queryEvent(event_type& event)
  {
    LOGPRINTF("%p ExecContext<raja>::queryEvent event %p\n", this, &event);
    return event.check();
  }

  void waitEvent(event_type& event)
  {
    LOGPRINTF("%p ExecContext<raja>::waitEvent event %p\n", this, &event);
    event.wait();
  }

  void destroyEvent(event_type& event)
  {
    LOGPRINTF("%p ExecContext<raja>::destroyEvent event %p\n", this, &event);
    event = event_type{};
  }

  template < typename body_type >
  void for_all(IdxT len, body_type&& body)
  {
    LOGPRINTF("%p ExecContext<raja>::for_all len %d\n", this, len);

    RAJA::TypedRangeSegment<IdxT> seg(0, len);

    RAJA::forall<for_all_policy>(base::res_launch(), seg, body);
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT len0, IdxT len1, body_type&& body)
  {
    LOGPRINTF("%p ExecContext<raja>::for_all_2d len0 %d len1 %d\n", this, len0, len1);

    RAJA::TypedRangeSegment<IdxT> seg0(0, len0);
    RAJA::TypedRangeSegment<IdxT> seg1(0, len1);

    RAJA::kernel_resource<for_all_2d_policy>(
        RAJA::make_tuple(seg0, seg1), base::res_launch(), body);
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT len0, IdxT len1, IdxT len2, body_type&& body)
  {
    LOGPRINTF("%p ExecContext<raja>::for_all_3d len0 %d len1 %d len2 %d\n", this, len0, len1, len2);

    RAJA::TypedRangeSegment<IdxT> seg0(0, len0);
    RAJA::TypedRangeSegment<IdxT> seg1(0, len1);
    RAJA::TypedRangeSegment<IdxT> seg2(0, len2);

    RAJA::kernel_resource<for_all_3d_policy>(
        RAJA::make_tuple(seg0, seg1, seg2), base::res_launch(), body);
    // base::synchronize();
  }

};


namespace detail
{

template < typename base_policy >
struct FuserStorage<ExecContext<raja_pol<base_policy>>>
{
  using context_type = ExecContext<raja_pol<base_policy>>;

  using workgroup_policy = typename context_type::fused_policy;
  using workgroup_allocator = typename context_type::fused_allocator;

  using workpool = RAJA::WorkPool< workgroup_policy,
                                   int,
                                   RAJA::xargs<>,
                                   workgroup_allocator >;

  using workgroup = RAJA::WorkGroup< workgroup_policy,
                                     int,
                                     RAJA::xargs<>,
                                     workgroup_allocator >;

  using worksite = RAJA::WorkSite< workgroup_policy,
                                   int,
                                   RAJA::xargs<>,
                                   workgroup_allocator >;

  struct WorkObjects
  {
    WorkObjects(context_type& con)
      : m_pool(workgroup_allocator{&con.util_aloc})
      , m_group(m_pool.instantiate())
      , m_site(m_group.run())
    {
      LOGPRINTF("%p FuserStorage<raja>::WorkObjects::WorkObjects\n", this);
    }

    ~WorkObjects()
    {
      LOGPRINTF("%p FuserStorage<raja>::WorkObjects::~WorkObjects\n", this);
    }

    workpool  m_pool;
    workgroup m_group;
    worksite  m_site;
  };

  using workgroups_list_type = std::list<WorkObjects>;
  using workgroups_iterator_type = typename workgroups_list_type::iterator;

  workgroups_list_type m_workGroups_list;
  workgroups_iterator_type m_workGroup_it;

  // note that these numbers are missing a factor of m_num_vars
  IdxT m_num_fused_loops_enqueued = 0;
  IdxT m_num_fused_loops_executed = 0;
  IdxT m_num_fused_loops_total    = 0;

  // vars for fused loops, stored in backend accessible memory
  std::vector<DataT*> m_variables;

  void allocate(context_type& con, std::vector<DataT*> const& variables, IdxT num_loops)
  {
    LOGPRINTF("%p FuserStorage<raja>::allocate num_loops %d\n", this, num_loops);
    if (m_variables.empty()) {

      m_num_fused_loops_enqueued = 0;
      m_num_fused_loops_executed = 0;
      m_num_fused_loops_total    = num_loops;

      // allocate per variable vars
      m_variables = variables;

      m_workGroup_it = m_workGroups_list.begin();
      LOGPRINTF("%p FuserStorage<raja>::allocate num_vars %zu\n", this, m_variables.size());
    }
  }

  void enqueue(context_type& con)
  {
    if (this->m_workGroup_it == this->m_workGroups_list.end()) {
      this->m_workGroups_list.emplace_back(con);
      this->m_workGroup_it = --this->m_workGroups_list.end();
      LOGPRINTF("%p FuserStorage<raja>::enqueue WorkObjects %p\n", this, &*this->m_workGroup_it);
    }
  }

  void exec(context_type& con)
  {
    LOGPRINTF("%p FuserStorage<raja>::exec WorkObjects %p\n", this, &*this->m_workGroup_it);
    this->m_workGroup_it->m_group = this->m_workGroup_it->m_pool.instantiate();
    this->m_workGroup_it->m_site  = this->m_workGroup_it->m_group.run(/*con.res_launch()*/);
    this->m_num_fused_loops_executed = this->m_num_fused_loops_enqueued;
    ++this->m_workGroup_it;
  }

  // There is potentially a race condition on these buffers as the allocations
  // are released back into the pool and could be used for another fuser
  // before this fuser is done executing.
  // This is safe for messages because there is synchronization between the
  // allocations (irecv, isend) and deallocations (wait_recv, wait_send).
  void deallocate(context_type& con)
  {
    LOGPRINTF("%p FuserStorage<raja>::deallocate \n", this);
    if (!this->m_variables.empty() && this->m_num_fused_loops_executed == this->m_num_fused_loops_total) {

      LOGPRINTF("%p FuserStorage<raja>::deallocate clear\n", this);
      // deallocate per variable vars
      this->m_variables.clear();

      for (WorkObjects& wo : this->m_workGroups_list) {
        wo.m_site.clear();
        wo.m_group.clear();
        wo.m_pool.clear();
      }
    }
  }
};

template < typename base_policy >
struct FuserPacker<ExecContext<raja_pol<base_policy>>> : FuserStorage<ExecContext<raja_pol<base_policy>>>
{
  using base = FuserStorage<ExecContext<raja_pol<base_policy>>>;
  using context_type = typename base::context_type;

  void allocate(context_type& con, std::vector<DataT*> const& variables, IdxT num_loops)
  {
    LOGPRINTF("%p FuserPacker<raja>::allocate num_loops %d\n", this, num_loops);
    if (this->m_variables.empty()) {
      base::allocate(con, variables, num_loops);
    }
  }

  // enqueue packing loops for all variables
  void enqueue(context_type& con, DataT* buf, LidxT const* indices, const IdxT nitems)
  {
    LOGPRINTF("%p FuserPacker<raja>::enqueue buf %p indices %p nitems %i\n", this, buf, indices, nitems);
    base::enqueue(con);

    RAJA::TypedRangeSegment<IdxT> seg(0, nitems);
    for (DataT const* src : this->m_variables) {
      LOGPRINTF("%p FuserPacker<raja>::enqueue enqueue %p buf %p[i] = src %p[indices %p] nitems %i\n", this, &*this->m_workGroup_it, buf, src, indices, nitems);
      this->m_workGroup_it->m_pool.enqueue(seg, make_copy_idxr_idxr(src, detail::indexer_list_i{indices},
                                                                    buf, detail::indexer_i{}));
      buf += nitems;
    }
    this->m_num_fused_loops_enqueued += 1;
  }

  void exec(context_type& con)
  {
    LOGPRINTF("%p FuserPacker<raja>::exec\n", this);
    base::exec(con);
  }

  void deallocate(context_type& con)
  {
    LOGPRINTF("%p FuserPacker<raja>::deallocate\n", this);
    if (!this->m_variables.empty() && this->m_num_fused_loops_executed == this->m_num_fused_loops_total) {
      base::deallocate(con);
    }
  }
};

template < typename base_policy >
struct FuserUnpacker<ExecContext<raja_pol<base_policy>>> : FuserStorage<ExecContext<raja_pol<base_policy>>>
{
  using base = FuserStorage<ExecContext<raja_pol<base_policy>>>;
  using context_type = typename base::context_type;

  void allocate(context_type& con, std::vector<DataT*> const& variables, IdxT num_loops)
  {
    LOGPRINTF("%p FuserUnpacker<raja>::allocate num_loops %d\n", this, num_loops);
    if (this->m_variables.empty()) {
      base::allocate(con, variables, num_loops);
    }
  }

  // enqueue unpacking loops for all variables
  void enqueue(context_type& con, DataT const* buf, LidxT const* indices, const IdxT nitems)
  {
    LOGPRINTF("%p FuserUnpacker<raja>::enqueue buf %p indices %p nitems %i\n", this, buf, indices, nitems);
    base::enqueue(con);

    RAJA::TypedRangeSegment<IdxT> seg(0, nitems);
    for (DataT* dst : this->m_variables) {
      LOGPRINTF("%p FuserUnpacker<raja>::enqueue enqueue %p dst %p[indices %p] = buf %p[i] nitems %i\n", this, &*this->m_workGroup_it, dst, indices, buf, nitems);
      this->m_workGroup_it->m_pool.enqueue(seg, make_copy_idxr_idxr(buf, detail::indexer_i{},
                                                                    dst, detail::indexer_list_i{indices}));
      buf += nitems;
    }
    this->m_num_fused_loops_enqueued += 1;
  }

  void exec(context_type& con)
  {
    LOGPRINTF("%p FuserUnpacker<raja>::exec\n", this);
    base::exec(con);
  }

  void deallocate(context_type& con)
  {
    LOGPRINTF("%p FuserUnpacker<raja>::deallocate\n", this);
    if (!this->m_variables.empty() && this->m_num_fused_loops_executed == this->m_num_fused_loops_total) {
      base::deallocate(con);
    }
  }
};

}

#endif // COMB_ENABLE_RAJA

#endif // _POL_RAJA_HPP
