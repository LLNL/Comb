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

#ifndef _COMB_HPP
#define _COMB_HPP

#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cctype>
#include <unistd.h>
#include <sched.h>

#include "memory.hpp"
#include "for_all.hpp"
#include "comm.hpp"
#include "profiling.hpp"
#include "utils.hpp"
#include "SetReset.hpp"
#include "MeshInfo.hpp"
#include "MeshData.hpp"

namespace COMB {

namespace detail {

  struct set_copy {
     DataT* data;
     const DataT* other;
     set_copy(DataT* data_, const DataT* other_) : data(data_), other(other_) {}
     COMB_HOST COMB_DEVICE
     void operator()(IdxT i, IdxT) const {
       IdxT zone = i;
       DataT next = other[zone];
       // FGPRINTF(FileGroup::proc, "%p[%i] = %f\n", data, zone, next);
       data[zone] = next;
     }
  };

  struct set_0 {
     DataT* data;
     set_0(DataT* data_) : data(data_) {}
     COMB_HOST COMB_DEVICE
     void operator()(IdxT i, IdxT) const {
       IdxT zone = i;
       DataT next = 0.0;
       // FGPRINTF(FileGroup::proc, "%p[%i] = %f\n", data, zone, next);
       data[zone] = next;
     }
  };

  struct set_n1 {
     DataT* data;
     set_n1(DataT* data_) : data(data_) {}
     COMB_HOST COMB_DEVICE
     void operator()(IdxT i, IdxT) const {
       IdxT zone = i;
       DataT next = -1.0;
       // FGPRINTF(FileGroup::proc, "%p[%i] = %f\n", data, zone, next);
       data[zone] = next;
     }
  };

  struct set_1 {
     IdxT ilen, ijlen;
     DataT* data;
     set_1(IdxT ilen_, IdxT ijlen_, DataT* data_) : ilen(ilen_), ijlen(ijlen_), data(data_) {}
     COMB_HOST COMB_DEVICE
     void operator()(IdxT k, IdxT j, IdxT i, IdxT idx) const {
       COMB::ignore_unused(idx);
       IdxT zone = i + j * ilen + k * ijlen;
       DataT next = 1.0;
       // FGPRINTF(FileGroup::proc, "%p[%i] = %f\n", data, zone, next);
       data[zone] = next;
     }
  };

  struct reset_1 {
     IdxT ilen, ijlen;
     DataT* data;
     IdxT imin, jmin, kmin;
     IdxT imax, jmax, kmax;
     reset_1(IdxT ilen_, IdxT ijlen_, DataT* data_, IdxT imin_, IdxT jmin_, IdxT kmin_, IdxT imax_, IdxT jmax_, IdxT kmax_)
       : ilen(ilen_), ijlen(ijlen_), data(data_)
       , imin(imin_), jmin(jmin_), kmin(kmin_)
       , imax(imax_), jmax(jmax_), kmax(kmax_)
     {}
     COMB_HOST COMB_DEVICE
     void operator()(IdxT k, IdxT j, IdxT i, IdxT idx) const {
       COMB::ignore_unused(idx);
       IdxT zone = i + j * ilen + k * ijlen;
       // DataT expected, found, next;
       // if (k >= kmin && k < kmax &&
       //     j >= jmin && j < jmax &&
       //     i >= imin && i < imax) {
       //   expected = 1.0; found = data[zone]; next = 1.0;
       // } else {
       //   expected = 0.0; found = data[zone]; next = -1.0;
       // }
       // if (found != expected) {
       //   FGPRINTF(FileGroup::proc, "zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
       // }
       //FGPRINTF(FileGroup::proc, "%p[%i] = %f\n", data, zone, 1.0);
       DataT next = 1.0;
       data[zone] = next;
     }
  };


} // namespace detail

extern void print_timer(CommInfo& comminfo, Timer& tm, const char* prefix = "");

extern void warmup(COMB::ExecContexts& exec,
                   COMB::Allocators& alloc,
                   Timer& tm, IdxT num_vars, IdxT len);

extern void test_copy(CommInfo& comminfo,
                      COMB::ExecContexts& exec,
                      COMB::Allocators& alloc,
                      COMB::ExecutorsAvailable& exec_avail,
                      Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats);

extern void test_cycles_mock(CommInfo& comminfo, MeshInfo& info,
                             COMB::ExecContexts& exec,
                             COMB::Allocators& alloc,
                             COMB::ExecutorsAvailable& exec_avail,
                             IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total);

extern void test_cycles_mpi(CommInfo& comminfo, MeshInfo& info,
                            COMB::ExecContexts& exec,
                            COMB::Allocators& alloc,
                            COMB::ExecutorsAvailable& exec_avail,
                            IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total);

#ifdef COMB_ENABLE_GPUMP
extern void test_cycles_gpump(CommInfo& comminfo, MeshInfo& info,
                              COMB::ExecContexts& exec,
                              COMB::Allocators& alloc,
                              COMB::ExecutorsAvailable& exec_avail,
                              IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total);
#endif

#ifdef COMB_ENABLE_MP
extern void test_cycles_mp(CommInfo& comminfo, MeshInfo& info,
                           COMB::ExecContexts& exec,
                           COMB::Allocators& alloc,
                           COMB::ExecutorsAvailable& exec_avail,
                           IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total);
#endif

#ifdef COMB_ENABLE_UMR
extern void test_cycles_umr(CommInfo& comminfo, MeshInfo& info,
                            COMB::ExecContexts& exec,
                            COMB::Allocators& alloc,
                            COMB::ExecutorsAvailable& exec_avail,
                            IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total);
#endif

} // namespace COMB

#endif // _COMB_HPP

