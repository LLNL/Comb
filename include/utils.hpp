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

#ifndef _UTILS_HPP
#define _UTILS_HPP

#include "config.hpp"

#include "print.hpp"

#include <cassert>
#include <cstdio>

using IdxT = int;
using LidxT = int;
using DataT = double;


namespace detail {

// std::exchange
// taken from https://en.cppreference.com/w/cpp/utility/exchange
// license http://creativecommons.org/licenses/by-sa/3.0/
template < typename T, typename U = T >
T exchange(T& obj, U&& new_value)
{
  T old_value = std::move(obj);
  obj = std::forward<U>(new_value);
  return old_value;
}

template < typename T, typename ... types >
struct Count;

template < typename T >
struct Count<T> {
  static const size_t value = 0;
};

template < typename T, typename ... types >
struct Count<T, T, types...> {
  static const size_t value = 1 + Count<T, types...>::value;
};

template < typename T, typename T0, typename ... types >
struct Count<T, T0, types...> {
  static const size_t value = Count<T, types...>::value;
};

struct indexer_kji {
  IdxT ijlen, ilen;
  indexer_kji(IdxT ijlen_, IdxT ilen_) : ijlen(ijlen_), ilen(ilen_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT j, IdxT i, IdxT) const { return i + j * ilen + k * ijlen; }
};
struct indexer_ji {
  IdxT ilen, koff;
  indexer_ji(IdxT ilen_, IdxT koff_) : ilen(ilen_), koff(koff_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT j, IdxT i, IdxT) const { return i + j * ilen + koff; }
};
struct indexer_ki {
  IdxT ijlen, joff;
  indexer_ki(IdxT ijlen_, IdxT joff_) : ijlen(ijlen_), joff(joff_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT i, IdxT) const { return i + joff + k * ijlen; }
};
struct indexer_kj {
  IdxT ijlen, ilen, ioff;
  indexer_kj(IdxT ijlen_, IdxT ilen_, IdxT ioff_) : ijlen(ijlen_), ilen(ilen_), ioff(ioff_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT j, IdxT) const { return ioff + j * ilen + k * ijlen; }
};

struct indexer_i {
  IdxT off;
  indexer_i(IdxT off_) : off(off_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT i, IdxT) const { return i + off; }
};
struct indexer_j {
  IdxT ilen, off;
  indexer_j(IdxT ilen_, IdxT off_) : ilen(ilen_), off(off_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT j, IdxT) const { return j * ilen + off; }
};
struct indexer_k {
  IdxT ijlen, off;
  indexer_k(IdxT ijlen_, IdxT off_) : ijlen(ijlen_), off(off_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT) const { return k * ijlen + off; }
};

struct indexer_ {
  IdxT off;
  indexer_(IdxT off_) : off(off_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT) const { return off; }
};

struct indexer_idx {
  indexer_idx() {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT idx) const { return idx; }
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT, IdxT idx) const { return idx; }
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT, IdxT, IdxT idx) const { return idx; }
};

struct indexer_list_idx {
  LidxT const* indices;
  indexer_list_idx(LidxT const* indices_) : indices(indices_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT idx) const { return indices[idx]; }
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT, IdxT idx) const { return indices[idx]; }
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT, IdxT, IdxT idx) const { return indices[idx]; }
};

template < typename T_src, typename I_src, typename T_dst, typename I_dst >
struct copy_idxr_idxr {
  T_src const* ptr_src;
  T_dst* ptr_dst;
  I_src idxr_src;
  I_dst idxr_dst;
  copy_idxr_idxr(T_src const* const& ptr_src_, I_src const& idxr_src_, T_dst* const& ptr_dst_, I_dst const& idxr_dst_) : ptr_src(ptr_src_), ptr_dst(ptr_dst_), idxr_src(idxr_src_), idxr_dst(idxr_dst_) {}
  template < typename ... Ts >
  COMB_HOST COMB_DEVICE void operator()(Ts... args) const
  {
    IdxT dst_i = idxr_dst(args...);
    IdxT src_i = idxr_src(args...);
    // FGPRINTF(FileGroup::proc, "copy_idxr_idxr %p[%i]{%f} = %p[%i]{%f} (%i)%i\n", ptr_dst, dst_i, (double)ptr_dst[dst_i],
    //                                                                    ptr_src, src_i, (double)ptr_src[src_i], args...);
    ptr_dst[dst_i] = ptr_src[src_i];
  }
};

template < typename T_src, typename I_src, typename T_dst, typename I_dst >
copy_idxr_idxr<T_src, I_src, T_dst, I_dst> make_copy_idxr_idxr(T_src* const& ptr_src, I_src const& idxr_src, T_dst* const& ptr_dst, I_dst const& idxr_dst) {
  return copy_idxr_idxr<T_src, I_src, T_dst, I_dst>(ptr_src, idxr_src, ptr_dst, idxr_dst);
}

template < typename I_src, typename T_dst, typename I_dst >
struct set_idxr_idxr {
  T_dst* ptr_dst;
  I_src idxr_src;
  I_dst idxr_dst;
  set_idxr_idxr(I_src const& idxr_src_, T_dst* const& ptr_dst_, I_dst const& idxr_dst_)
    : ptr_dst(ptr_dst_)
    , idxr_src(idxr_src_)
    , idxr_dst(idxr_dst_)
  { }
  template < typename ... Ts >
  COMB_HOST COMB_DEVICE void operator()(Ts... args) const
  {
    IdxT dst_i = idxr_dst(args...);
    IdxT src_i = idxr_src(args...);
    // FGPRINTF(FileGroup::proc, "set_idxr_idxr %p[%i]{%f} = %i (%i %i %i)%i\n", ptr_dst, dst_i, (double)ptr_dst[dst_i], src_i, args...);
    ptr_dst[dst_i] = src_i;
  }
};

template < typename I_src, typename T_dst, typename I_dst >
set_idxr_idxr<I_src, T_dst, I_dst> make_set_idxr_idxr(I_src const& idxr_src, T_dst* const& ptr_dst, I_dst const& idxr_dst) {
  return set_idxr_idxr<I_src, T_dst, I_dst>(idxr_src, ptr_dst, idxr_dst);
}

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

#endif // _UTILS_HPP

