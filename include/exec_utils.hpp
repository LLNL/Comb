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
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT j, IdxT i) const { return i + j * ilen + k * ijlen; }
};
struct indexer_ji {
  IdxT ilen;
  indexer_ji(IdxT ilen_) : ilen(ilen_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT j, IdxT i) const { return i + j * ilen; }
};
struct indexer_i {
  indexer_i() {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT i) const { return i; }
};

struct indexer_offset_kji {
  IdxT ijlen, ilen;
  IdxT imin, jmin, kmin;
  indexer_offset_kji(IdxT kmin_, IdxT jmin_, IdxT imin_, IdxT ijlen_, IdxT ilen_)
    : ijlen(ijlen_), ilen(ilen_)
    , imin(imin_), jmin(jmin_), kmin(kmin_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT j, IdxT i) const { return (i+imin) + (j+jmin) * ilen + (k+kmin) * ijlen; }
};

struct indexer_list_kji {
  LidxT const* indices;
  IdxT ijlen, ilen;
  indexer_list_kji(LidxT const* indices_, IdxT ijlen_, IdxT ilen_) : indices(indices_), ijlen(ijlen_), ilen(ilen_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT j, IdxT i) const { return indices[i + j * ilen + k * ijlen]; }
};
struct indexer_list_ji {
  LidxT const* indices;
  IdxT ilen;
  indexer_list_ji(LidxT const* indices_, IdxT ilen_) : indices(indices_), ilen(ilen_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT j, IdxT i) const { return indices[i + j * ilen]; }
};
struct indexer_list_i {
  LidxT const* indices;
  indexer_list_i(LidxT const* indices_) : indices(indices_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT i) const { return indices[i]; }
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

} // namespace detail

#endif // _UTILS_HPP

