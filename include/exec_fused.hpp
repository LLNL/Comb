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
  void operator() (IdxT, IdxT idx) const
  {
    IdxT i0 = idx / len1;
    IdxT i1 = idx - i0 * len1;

    //FGPRINTF(FileGroup::proc, "adapter_2d (%i+%i %i+%i)%i\n", i0, begin0, i1, begin1, idx);
    //assert(0 <= i0 + begin0 && i0 + begin0 < 3);
    //assert(0 <= i1 + begin1 && i1 + begin1 < 3);

    body(i0 + begin0, i1 + begin1, idx);
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
  void operator() (IdxT, IdxT idx) const
  {
    IdxT i0 = idx / len12;
    IdxT idx12 = idx - i0 * len12;

    IdxT i1 = idx12 / len2;
    IdxT i2 = idx12 - i1 * len2;

    //FGPRINTF(FileGroup::proc, "adapter_3d (%i+%i %i+%i %i+%i)%i\n", i0, begin0, i1, begin1, i2, begin2, idx);
    //assert(0 <= i0 + begin0 && i0 + begin0 < 3);
    //assert(0 <= i1 + begin1 && i1 + begin1 < 3);
    //assert(0 <= i2 + begin2 && i2 + begin2 < 13);

    body(i0 + begin0, i1 + begin1, i2 + begin2, idx);
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
  void operator()(IdxT i, IdxT)
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
  void operator()(IdxT i, IdxT)
  {
    // if (i == 0) {
    //   FGPRINTF(FileGroup::proc, "fused_packer buf %p, dst %p, idx %p, len %i\n", buf, dst, idx, len); FFLUSH(stdout);
    // }
    dst[idx[i]] = buf[i];
  }
};

} // namespace detail


namespace COMB {

} // namespace COMB

#endif // _FUSED_HPP
