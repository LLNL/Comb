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

#ifndef _FUSED_HPP
#define _FUSED_HPP

#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <type_traits>

#include "memory.hpp"
#include "ExecContext.hpp"
#include "exec_utils.hpp"


inline bool& comb_allow_per_message_pack_fusing()
{
  static bool allow = true;
  return allow;
}

inline bool& comb_allow_pack_loop_fusion()
{
  static bool allow = true;
  return allow;
}

namespace detail {

template < typename body_type >
struct adapter_2d {
  IdxT begin0, begin1;
  IdxT len1;
  body_type body;
  template < typename body_type_ >
  adapter_2d(IdxT begin0_, IdxT end0_, IdxT begin1_, IdxT end1_, body_type_&& body_)
    : begin0(begin0_)
    , begin1(begin1_)
    , len1(end1_ - begin1_)
    , body(std::forward<body_type_>(body_))
  { COMB::ignore_unused(end0_); }
  COMB_HOST COMB_DEVICE
  void operator() (IdxT i) const
  {
    IdxT i0 = i / len1;
    IdxT i1 = i - i0 * len1;

    //FGPRINTF(FileGroup::proc, "adapter_2d (%i+%i %i+%i)%i\n", i0, begin0, i1, begin1, i);
    //assert(0 <= i0 + begin0 && i0 + begin0 < 3);
    //assert(0 <= i1 + begin1 && i1 + begin1 < 3);

    body(i0 + begin0, i1 + begin1);
  }
};

template < typename body_type >
struct adapter_3d {
  IdxT begin0, begin1, begin2;
  IdxT len2, len12;
  body_type body;
  template < typename body_type_ >
  adapter_3d(IdxT begin0_, IdxT end0_, IdxT begin1_, IdxT end1_, IdxT begin2_, IdxT end2_, body_type_&& body_)
    : begin0(begin0_)
    , begin1(begin1_)
    , begin2(begin2_)
    , len2(end2_ - begin2_)
    , len12((end1_ - begin1_) * (end2_ - begin2_))
    , body(std::forward<body_type_>(body_))
  { COMB::ignore_unused(end0_); }
  COMB_HOST COMB_DEVICE
  void operator() (IdxT i) const
  {
    IdxT i0 = i / len12;
    IdxT idx12 = i - i0 * len12;

    IdxT i1 = idx12 / len2;
    IdxT i2 = idx12 - i1 * len2;

    //FGPRINTF(FileGroup::proc, "adapter_3d (%i+%i %i+%i %i+%i)%i\n", i0, begin0, i1, begin1, i2, begin2, i);
    //assert(0 <= i0 + begin0 && i0 + begin0 < 3);
    //assert(0 <= i1 + begin1 && i1 + begin1 < 3);
    //assert(0 <= i2 + begin2 && i2 + begin2 < 13);

    body(i0 + begin0, i1 + begin1, i2 + begin2);
  }
};

struct fused_packer
{
  DataT const** srcs;
  DataT**       bufs;
  LidxT const** idxs;
  IdxT const*   lens;

  DataT const* src = nullptr;
  DataT*       bufk = nullptr;
  DataT*       buf = nullptr;
  LidxT const* idx = nullptr;
  IdxT         len = 0;

  fused_packer(DataT const** srcs_, DataT** bufs_, LidxT const** idxs_, IdxT const* lens_)
    : srcs(srcs_)
    , bufs(bufs_)
    , idxs(idxs_)
    , lens(lens_)
  { }

  COMB_HOST COMB_DEVICE
  void set_outer(IdxT k)
  {
    len = lens[k];
    idx = idxs[k];
    bufk = bufs[k];
  }

  COMB_HOST COMB_DEVICE
  void set_inner(IdxT j)
  {
    src = srcs[j];
    buf = bufk + j*len;
  }

  // must be run for all i in [0, len)
  COMB_HOST COMB_DEVICE
  void operator()(IdxT i)
  {
    // if (i == 0) {
    //   FGPRINTF(FileGroup::proc, "fused_packer buf %p, src %p, idx %p, len %i\n", buf, src, idx, len); FFLUSH(stdout);
    // }
    buf[i] = src[idx[i]];
  }
};

struct fused_unpacker
{
  DataT**       dsts;
  DataT const** bufs;
  LidxT const** idxs;
  IdxT  const*  lens;

  DataT*       dst = nullptr;
  DataT const* bufk = nullptr;
  DataT const* buf = nullptr;
  LidxT const* idx = nullptr;
  IdxT         len = 0;

  fused_unpacker(DataT** dsts_, DataT const** bufs_, LidxT const** idxs_, IdxT const* lens_)
    : dsts(dsts_)
    , bufs(bufs_)
    , idxs(idxs_)
    , lens(lens_)
  { }

  COMB_HOST COMB_DEVICE
  void set_outer(IdxT k)
  {
    len = lens[k];
    idx = idxs[k];
    bufk = bufs[k];
  }

  COMB_HOST COMB_DEVICE
  void set_inner(IdxT j)
  {
    dst = dsts[j];
    buf = bufk + j*len;
  }

  // must be run for all i in [0, len)
  COMB_HOST COMB_DEVICE
  void operator()(IdxT i)
  {
    // if (i == 0) {
    //   FGPRINTF(FileGroup::proc, "fused_packer buf %p, dst %p, idx %p, len %i\n", buf, dst, idx, len); FFLUSH(stdout);
    // }
    dst[idx[i]] = buf[i];
  }
};


template < typename context_type >
struct FuserStorage
{
  // note that these numbers are missing a factor of m_num_vars
  IdxT m_num_fused_iterations = 0;
  IdxT m_num_fused_loops_enqueued = 0;
  IdxT m_num_fused_loops_executed = 0;
  IdxT m_num_fused_loops_total = 0;

  // vars for fused loops, stored in backend accessible memory
  DataT** m_vars = nullptr;
  IdxT m_num_vars = 0;

  LidxT const** m_idxs = nullptr;
  IdxT*         m_lens = nullptr;


  DataT      ** get_dsts() { return                m_vars; }
  DataT const** get_srcs() { return (DataT const**)m_vars; }

  void allocate(context_type& con, std::vector<DataT*> const& variables, IdxT num_loops)
  {
    if (m_vars == nullptr) {

      m_num_fused_iterations     = 0;
      m_num_fused_loops_enqueued = 0;
      m_num_fused_loops_executed = 0;
      m_num_fused_loops_total    = num_loops;

      // allocate per variable vars
      m_num_vars = variables.size();
      m_vars = (DataT**)con.util_aloc.allocate(m_num_vars*sizeof(DataT const*));

      // variable vars initialized here
      for (IdxT i = 0; i < m_num_vars; ++i) {
        m_vars[i] = variables[i];
      }

      // allocate per item vars
      m_idxs = (LidxT const**)con.util_aloc.allocate(num_loops*sizeof(LidxT const*));
      m_lens = (IdxT*)        con.util_aloc.allocate(num_loops*sizeof(IdxT));

      // item vars initialized in pack
    }
  }

  // There is potentially a race condition on these buffers as the allocations
  // are released back into the pool and could be used for another fuser
  // before this fuser is done executing.
  // This is safe for messages because there is synchronization between the
  // allocations (irecv, isend) and deallocations (wait_recv, wait_send).
  void deallocate(context_type& con)
  {
    if (m_vars != nullptr && this->m_num_fused_loops_executed == this->m_num_fused_loops_total) {

      // deallocate per variable vars
      con.util_aloc.deallocate(m_vars); m_vars = nullptr;
      m_num_vars = 0;

      // deallocate per item vars
      con.util_aloc.deallocate(m_idxs); m_idxs = nullptr;
      con.util_aloc.deallocate(m_lens); m_lens = nullptr;
    }
  }
};

template < typename context_type >
struct FuserPacker : FuserStorage<context_type>
{
  using base = FuserStorage<context_type>;

  DataT** m_bufs = nullptr;

  void allocate(context_type& con, std::vector<DataT*> const& variables, IdxT num_items)
  {
    if (this->m_vars == nullptr) {
      base::allocate(con, variables, num_items);

      this->m_bufs = (DataT**)con.util_aloc.allocate(num_items*sizeof(DataT*));
    }
  }

  // enqueue packing loops for all variables
  void enqueue(context_type& /*con*/, DataT* buf, LidxT const* indices, const IdxT nitems)
  {
    this->m_bufs[this->m_num_fused_loops_enqueued] = buf;
    this->m_idxs[this->m_num_fused_loops_enqueued] = indices;
    this->m_lens[this->m_num_fused_loops_enqueued] = nitems;
    this->m_num_fused_iterations += nitems;
    this->m_num_fused_loops_enqueued += 1;
  }

  void exec(context_type& con)
  {
    IdxT num_fused_loops = this->m_num_fused_loops_enqueued - this->m_num_fused_loops_executed;
    IdxT avg_iterations = (this->m_num_fused_iterations + num_fused_loops - 1) / num_fused_loops;
    con.fused(num_fused_loops, this->m_num_vars, avg_iterations,
        fused_packer(this->get_srcs(), this->m_bufs+this->m_num_fused_loops_executed,
                                       this->m_idxs+this->m_num_fused_loops_executed,
                                       this->m_lens+this->m_num_fused_loops_executed));
    this->m_num_fused_iterations = 0;
    this->m_num_fused_loops_executed = this->m_num_fused_loops_enqueued;
  }

  void deallocate(context_type& con)
  {
    if (this->m_vars != nullptr && this->m_num_fused_loops_executed == this->m_num_fused_loops_total) {
      base::deallocate(con);

      con.util_aloc.deallocate(this->m_bufs); this->m_bufs = nullptr;
    }
  }
};

template < typename context_type >
struct FuserUnpacker : FuserStorage<context_type>
{
  using base = FuserStorage<context_type>;

  DataT const** m_bufs = nullptr;

  void allocate(context_type& con, std::vector<DataT*> const& variables, IdxT num_items)
  {
    if (this->m_vars == nullptr) {
      base::allocate(con, variables, num_items);

      this->m_bufs = (DataT const**)con.util_aloc.allocate(num_items*sizeof(DataT const*));
    }
  }

  // enqueue unpacking loops for all variables
  void enqueue(context_type& /*con*/, DataT const* buf, LidxT const* indices, const IdxT nitems)
  {
    this->m_bufs[this->m_num_fused_loops_enqueued] = buf;
    this->m_idxs[this->m_num_fused_loops_enqueued] = indices;
    this->m_lens[this->m_num_fused_loops_enqueued] = nitems;
    this->m_num_fused_iterations += nitems;
    this->m_num_fused_loops_enqueued += 1;
  }

  void exec(context_type& con)
  {
    IdxT num_fused_loops = this->m_num_fused_loops_enqueued - this->m_num_fused_loops_executed;
    IdxT avg_iterations = (this->m_num_fused_iterations + num_fused_loops - 1) / num_fused_loops;
    con.fused(num_fused_loops, this->m_num_vars, avg_iterations,
        fused_unpacker(this->get_dsts(), this->m_bufs+this->m_num_fused_loops_executed,
                                         this->m_idxs+this->m_num_fused_loops_executed,
                                         this->m_lens+this->m_num_fused_loops_executed));
    this->m_num_fused_iterations = 0;
    this->m_num_fused_loops_executed = this->m_num_fused_loops_enqueued;
  }

  void deallocate(context_type& con)
  {
    if (this->m_vars != nullptr && this->m_num_fused_loops_executed == this->m_num_fused_loops_total) {
      base::deallocate(con);

      con.util_aloc.deallocate(this->m_bufs); this->m_bufs = nullptr;
    }
  }
};

} // namespace detail

#endif // _FUSED_HPP
