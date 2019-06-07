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

#ifndef _FOR_ALL_HPP
#define _FOR_ALL_HPP

#include "config.hpp"

#include <cstdio>
#include <cstdlib>

#include <type_traits>

#include "utils.hpp"
#include "memory.hpp"

namespace detail {

template < typename type >
struct Synchronizer {
  using value_type = type;
  value_type const& t;
  Synchronizer(value_type const& t_) : t(t_) {}
  void operator()() const { synchronize(t); }
};

template < typename type >
struct PersistentLauncher {
  using value_type = type;
  value_type const& t;
  PersistentLauncher(value_type const& t_) : t(t_) {}
  void operator()() const { persistent_launch(t); }
};

template < typename type >
struct BatchLauncher {
  using value_type = type;
  value_type const& t;
  BatchLauncher(value_type const& t_) : t(t_) {}
  void operator()() const { batch_launch(t); }
};

template < typename type >
struct PersistentStopper {
  using value_type = type;
  value_type const& t;
  PersistentStopper(value_type const& t_) : t(t_) {}
  void operator()() const { persistent_stop(t); }
};

template < typename type, size_t repeats >
struct ConditionalOperator {
  ConditionalOperator(typename type::value_type const&) {}
  void operator()() const { }
};

template < typename type >
struct ConditionalOperator<type, 0> : type {
  using parent = type;
  ConditionalOperator(typename type::value_type const& t_) : parent(t_) {}
  void operator()() const { parent::operator()(); }
};

template < typename ... types >
struct MultiOperator;

template < >
struct MultiOperator<> {
  void operator()() const { }
};

template < typename type0, typename ... types >
struct MultiOperator<type0, types...> : ConditionalOperator<type0, Count<type0, types...>::value>, MultiOperator<types...> {
  using cparent = ConditionalOperator<type0, Count<type0, types...>::value>;
  using mparent = MultiOperator<types...>;
  MultiOperator(typename type0::value_type const& t_, typename types::value_type const&... ts_) : cparent(t_), mparent(ts_...) {}
  void operator()() const { cparent::operator()(); mparent::operator()(); }
};

} // namespace detail

// multiple argument synchronization and other functions
template < typename policy0, typename policy1, typename... policies >
inline void synchronize(policy0 const& p0, policy1 const& p1, policies const&...ps)
{
  detail::MultiOperator<detail::Synchronizer<policy0>, detail::Synchronizer<policy1>, detail::Synchronizer<policies>...>{p0, p1, ps...}();
}

template < typename policy0, typename policy1, typename... policies >
inline void persistent_launch(policy0 const& p0, policy1 const& p1, policies const&...ps)
{
  detail::MultiOperator<detail::PersistentLauncher<policy0>, detail::PersistentLauncher<policy1>, detail::PersistentLauncher<policies>...>{p0, p1, ps...}();
}

template < typename policy0, typename policy1, typename... policies >
inline void batch_launch(policy0 const& p0, policy1 const& p1, policies const&...ps)
{
  detail::MultiOperator<detail::BatchLauncher<policy0>, detail::BatchLauncher<policy1>, detail::BatchLauncher<policies>...>{p0, p1, ps...}();
}

template < typename policy0, typename policy1, typename... policies >
inline void persistent_stop(policy0 const& p0, policy1 const& p1, policies const&...ps)
{
  detail::MultiOperator<detail::PersistentStopper<policy0>, detail::PersistentStopper<policy1>, detail::PersistentStopper<policies>...>{p0, p1, ps...}();
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

    //FPRINTF(stdout, "adapter_2d (%i+%i %i+%i)%i\n", i0, begin0, i1, begin1, idx);
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

    //FPRINTF(stdout, "adapter_3d (%i+%i %i+%i %i+%i)%i\n", i0, begin0, i1, begin1, i2, begin2, idx);
    //assert(0 <= i0 + begin0 && i0 + begin0 < 3);
    //assert(0 <= i1 + begin1 && i1 + begin1 < 3);
    //assert(0 <= i2 + begin2 && i2 + begin2 < 13);

    body(i0 + begin0, i1 + begin1, i2 + begin2, idx);
  }
};

} // namespace detail


#include "pol_seq.hpp"
#include "pol_omp.hpp"
#include "pol_cuda.hpp"
#include "pol_cuda_batch.hpp"
#include "pol_cuda_persistent.hpp"
#include "pol_cuda_graph.hpp"
#include "pol_mpi_type.hpp"

struct ExecutorsAvailable
{
  bool seq = false;
  bool omp = false;
  bool cuda = false;
  bool cuda_batch = false;
  bool cuda_batch_fewgs = false;
  bool cuda_persistent = false;
  bool cuda_persistent_fewgs = false;
  bool cuda_graph = false;
  bool mpi_type = false;
};

#endif // _FOR_ALL_HPP
