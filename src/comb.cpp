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

#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cctype>
#include <unistd.h>
#include <sched.h>

#include <mpi.h>

#ifdef COMB_ENABLE_OPENMP
#include <omp.h>
#endif

#include "memory.hpp"
#include "for_all.hpp"
#include "profiling.hpp"
#include "MeshInfo.hpp"
#include "MeshData.hpp"
#include "comm.hpp"
#include "CommFactory.hpp"
#include "SetReset.hpp"

#ifdef COMB_ENABLE_CUDA
#include "batch_utils.hpp"
#endif

#define PRINT_THREAD_MAP

#ifdef PRINT_THREAD_MAP
#include <linux/sched.h>
#endif

namespace detail {

  struct set_copy {
     DataT* data;
     const DataT* other;
     set_copy(DataT* data_, const DataT* other_) : data(data_), other(other_) {}
     COMB_HOST COMB_DEVICE
     void operator()(IdxT i, IdxT) const {
       IdxT zone = i;
       DataT next = other[zone];
       // FPRINTF(stdout, "%p[%i] = %f\n", data, zone, next);
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
       // FPRINTF(stdout, "%p[%i] = %f\n", data, zone, next);
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
       // FPRINTF(stdout, "%p[%i] = %f\n", data, zone, next);
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
       // FPRINTF(stdout, "%p[%i] = %f\n", data, zone, next);
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
       //   FPRINTF(stdout, "zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
       // }
       //FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
       DataT next = 1.0;
       data[zone] = next;
     }
  };

} // namespace detail

void print_timer(CommInfo& comm_info, Timer& tm, const char* prefix = "") {

  auto res = tm.getStats();

  int max_name_len = 0;

  for (auto& stat : res) {
    max_name_len = std::max(max_name_len, (int)stat.name.size());
  }

  double* sums = new double[res.size()];
  double* mins = new double[res.size()];
  double* maxs = new double[res.size()];
  long  * nums = new long  [res.size()];

  for (int i = 0; i < (int)res.size(); ++i) {
    sums[i] = res[i].sum;
    mins[i] = res[i].min;
    maxs[i] = res[i].max;
    nums[i] = res[i].num;
  }

  double* final_sums = nullptr;
  double* final_mins = nullptr;
  double* final_maxs = nullptr;
  long  * final_nums = nullptr;
  if (comm_info.rank == 0) {
    final_sums = new double[res.size()];
    final_mins = new double[res.size()];
    final_maxs = new double[res.size()];
    final_nums = new long  [res.size()];
  }

  MPI_Reduce(sums, final_sums, res.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(mins, final_mins, res.size(), MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(maxs, final_maxs, res.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(nums, final_nums, res.size(), MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);

  if (comm_info.rank == 0) {

    for (int i = 0; i < (int)res.size(); ++i) {
      int padding = max_name_len - res[i].name.size();
      comm_info.print(FileGroup::summary, "%s%s:%*s num %ld sum %.9f s min %.9f s max %.9f s\n",
                             prefix, res[i].name.c_str(), padding, "", final_nums[i], final_sums[i], final_mins[i], final_maxs[i]);
    }

    delete[] final_sums;
    delete[] final_mins;
    delete[] final_maxs;
    delete[] final_nums;
  }

  for (int i = 0; i < (int)res.size(); ++i) {
    int padding = max_name_len - res[i].name.size();
    comm_info.print(FileGroup::proc, "%s%s:%*s num %ld sum %.9f s min %.9f s max %.9f s\n",
                        prefix, res[i].name.c_str(), padding, "", nums[i], sums[i], mins[i], maxs[i]);
  }

  delete[] sums;
  delete[] mins;
  delete[] maxs;
  delete[] nums;
}

template < typename pol_loop, typename pol_many, typename pol_few >
void do_cycles(CommInfo& comm_info, MeshInfo& info, IdxT num_vars, IdxT ncycles, COMB::Allocator& aloc_mesh, COMB::Allocator& aloc_many, COMB::Allocator& aloc_few, Timer& tm, Timer& tm_total)
{
    tm_total.clear();
    tm.clear();

    char rname[1024] = ""; snprintf(rname, 1024, "Buffers %s %s %s %s", pol_many::get_name(), aloc_many.name(), pol_few::get_name(), aloc_few.name());
    char test_name[1024] = ""; snprintf(test_name, 1024, "Mesh %s %s %s", pol_loop::get_name(), aloc_mesh.name(), rname);
    comm_info.print(FileGroup::all, "Starting test %s\n", test_name);

    Range r0(test_name, Range::orange);

    Comm<pol_many, pol_few> comm(comm_info, aloc_mesh, aloc_many, aloc_few);

    comm.comminfo.barrier();

    tm_total.start("start-up");

    std::vector<MeshData> vars;
    vars.reserve(num_vars);

    {
      CommFactory factory(comm.comminfo);

      for (IdxT i = 0; i < num_vars; ++i) {

        vars.push_back(MeshData(info, aloc_mesh));

        vars[i].allocate();

        DataT* data = vars[i].data();
        IdxT totallen = info.totallen;

        for_all(pol_loop{}, 0, totallen,
                            detail::set_n1(data));

        factory.add_var(vars[i]);

        synchronize(pol_loop{});
      }

      factory.populate(comm);
    }

    tm_total.stop();

    comm.comminfo.barrier();

    Range r1("test comm", Range::indigo);

    tm_total.start("test-comm");

    { // test comm

      bool mock_communication = comm.comminfo.mock_communication;
      IdxT imin = info.min[0];
      IdxT jmin = info.min[1];
      IdxT kmin = info.min[2];
      IdxT imax = info.max[0];
      IdxT jmax = info.max[1];
      IdxT kmax = info.max[2];
      IdxT ilen = info.len[0];
      IdxT jlen = info.len[1];
      IdxT klen = info.len[2];
      IdxT iglobal_offset = info.global_offset[0];
      IdxT jglobal_offset = info.global_offset[1];
      IdxT kglobal_offset = info.global_offset[2];
      IdxT ilen_global = info.global.sizes[0];
      IdxT jlen_global = info.global.sizes[1];
      IdxT klen_global = info.global.sizes[2];
      IdxT iperiodic = info.global.periodic[0];
      IdxT jperiodic = info.global.periodic[1];
      IdxT kperiodic = info.global.periodic[2];
      IdxT ighost_width = info.ghost_widths[0];
      IdxT jghost_width = info.ghost_widths[1];
      IdxT kghost_width = info.ghost_widths[2];
      IdxT ijlen = info.stride[2];
      IdxT ijlen_global = ilen_global * jlen_global;


      Range r2("pre-comm", Range::red);
      // tm.start("pre-comm");

      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();
        IdxT var_i = i + 1;

        for_all_3d(pol_loop{}, 0, klen,
                               0, jlen,
                               0, ilen,
                               [=] COMB_HOST COMB_DEVICE (IdxT k, IdxT j, IdxT i, IdxT idx) {
          COMB::ignore_unused(idx);
          IdxT zone = i + j * ilen + k * ijlen;
          IdxT iglobal = i + iglobal_offset;
          if (iperiodic) {
            iglobal = iglobal % ilen_global;
            if (iglobal < 0) iglobal += ilen_global;
          }
          IdxT jglobal = j + jglobal_offset;
          if (jperiodic) {
            jglobal = jglobal % jlen_global;
            if (jglobal < 0) jglobal += jlen_global;
          }
          IdxT kglobal = k + kglobal_offset;
          if (kperiodic) {
            kglobal = kglobal % klen_global;
            if (kglobal < 0) kglobal += klen_global;
          }
          IdxT zone_global = iglobal + jglobal * ilen_global + kglobal * ijlen_global;
          DataT expected, found, next;
          int branchid = -1;
          if (k >= kmin+kghost_width && k < kmax-kghost_width &&
              j >= jmin+jghost_width && j < jmax-jghost_width &&
              i >= imin+ighost_width && i < imax-ighost_width) {
            // interior non-communicated zones
            expected = -1.0; found = data[zone]; next =-(zone_global+var_i);
            branchid = 0;
          } else if (k >= kmin && k < kmax &&
                     j >= jmin && j < jmax &&
                     i >= imin && i < imax) {
            // interior communicated zones
            expected = -1.0; found = data[zone]; next = zone_global + var_i;
            branchid = 1;
          } else if (iglobal < 0 || iglobal >= ilen_global ||
                     jglobal < 0 || jglobal >= jlen_global ||
                     kglobal < 0 || kglobal >= klen_global) {
            // out of global bounds exterior zones, some may be owned others not
            // some may be communicated if at least one dimension is periodic
            // and another is non-periodic
            expected = -1.0; found = data[zone]; next = zone_global + var_i;
            branchid = 2;
          } else {
            // in global bounds exterior zones
            expected = -1.0; found = data[zone]; next =-(zone_global+var_i);
            branchid = 3;
          }
          if (!mock_communication) {
            if (found != expected) {
              FPRINTF(stdout, "%p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
            }
            // FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
            assert(found == expected);
          }
          data[zone] = next;
        });
      }

      synchronize(pol_loop{});

      // tm.stop();
      r2.restart("post-recv", Range::pink);
      // tm.start("post-recv");

      comm.postRecv();

      // tm.stop();
      r2.restart("post-send", Range::pink);
      // tm.start("post-send");

      comm.postSend();

      // tm.stop();
      r2.stop();

      // for (IdxT i = 0; i < num_vars; ++i) {

      //   DataT* data = vars[i].data();
      //   IdxT var_i = i + 1;

      //   for_all_3d(pol_loop{}, 0, klen,
      //                          0, jlen,
      //                          0, ilen,
      //                          [=] COMB_HOST COMB_DEVICE (IdxT k, IdxT j, IdxT i, IdxT idx) {
      //     COMB::ignore_unused(idx);
      //     IdxT zone = i + j * ilen + k * ijlen;
      //     IdxT iglobal = i + iglobal_offset;
      //     if (iperiodic) {
      //       iglobal = iglobal % ilen_global;
      //       if (iglobal < 0) iglobal += ilen_global;
      //     }
      //     IdxT jglobal = j + jglobal_offset;
      //     if (jperiodic) {
      //       jglobal = jglobal % jlen_global;
      //       if (jglobal < 0) jglobal += jlen_global;
      //     }
      //     IdxT kglobal = k + kglobal_offset;
      //     if (kperiodic) {
      //       kglobal = kglobal % klen_global;
      //       if (kglobal < 0) kglobal += klen_global;
      //     }
      //     IdxT zone_global = iglobal + jglobal * ilen_global + kglobal * ijlen_global;
      //     DataT expected, found, next;
      //     int branchid = -1;
      //     if (k >= kmin+kghost_width && k < kmax-kghost_width &&
      //         j >= jmin+jghost_width && j < jmax-jghost_width &&
      //         i >= imin+ighost_width && i < imax-ighost_width) {
      //       // interior non-communicated zones should not have changed value
      //       expected =-(zone_global+var_i); found = data[zone]; next = -1.0;
      //       branchid = 0;
      //       if (!mock_communication) {
      //         if (found != expected) {
      //           FPRINTF(stdout, "%p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
      //         }
      //         // FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
      //         assert(found == expected);
      //       }
      //       data[zone] = next;
      //     }
      //     // other zones may be participating in communication, do not access
      //   });
      // }

      // synchronize(pol_loop{});


      r2.start("wait-recv", Range::pink);
      // tm.start("wait-recv");

      comm.waitRecv();

      // tm.stop();
      r2.restart("wait-send", Range::pink);
      // tm.start("wait-send");

      comm.waitSend();

      // tm.stop();
      r2.restart("post-comm", Range::red);
      // tm.start("post-comm");

      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();
        IdxT var_i = i + 1;

        for_all_3d(pol_loop{}, 0, klen,
                               0, jlen,
                               0, ilen,
                               [=] COMB_HOST COMB_DEVICE (IdxT k, IdxT j, IdxT i, IdxT idx) {
          COMB::ignore_unused(idx);
          IdxT zone = i + j * ilen + k * ijlen;
          IdxT iglobal = i + iglobal_offset;
          if (iperiodic) {
            iglobal = iglobal % ilen_global;
            if (iglobal < 0) iglobal += ilen_global;
          }
          IdxT jglobal = j + jglobal_offset;
          if (jperiodic) {
            jglobal = jglobal % jlen_global;
            if (jglobal < 0) jglobal += jlen_global;
          }
          IdxT kglobal = k + kglobal_offset;
          if (kperiodic) {
            kglobal = kglobal % klen_global;
            if (kglobal < 0) kglobal += klen_global;
          }
          IdxT zone_global = iglobal + jglobal * ilen_global + kglobal * ijlen_global;
          DataT expected, found, next;
          int branchid = -1;
          if (k >= kmin+kghost_width && k < kmax-kghost_width &&
              j >= jmin+jghost_width && j < jmax-jghost_width &&
              i >= imin+ighost_width && i < imax-ighost_width) {
            // interior non-communicated zones should not have changed value
            expected =-(zone_global+var_i); found = data[zone]; next = 1.0;
            branchid = 0;
          } else if (k >= kmin && k < kmax &&
                     j >= jmin && j < jmax &&
                     i >= imin && i < imax) {
            // interior communicated zones should not have changed value
            expected = zone_global + var_i; found = data[zone]; next = 1.0;
            branchid = 1;
          } else if (iglobal < 0 || iglobal >= ilen_global ||
                     jglobal < 0 || jglobal >= jlen_global ||
                     kglobal < 0 || kglobal >= klen_global) {
            // out of global bounds exterior zones should not have changed value
            // some may have been communicated, but values should be the same
            expected = zone_global + var_i; found = data[zone]; next = -1.0;
            branchid = 2;
          } else {
            // in global bounds exterior zones should have changed value
            // should now be populated with data from another rank
            expected = zone_global + var_i; found = data[zone]; next = -1.0;
            branchid = 3;
          }
          if (!mock_communication) {
            if (found != expected) {
              FPRINTF(stdout, "%p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
            }
            // FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
            assert(found == expected);
          }
          data[zone] = next;
        });
      }

      synchronize(pol_loop{});

      // tm.stop();
      r2.stop();
    }

    comm.comminfo.barrier();

    tm_total.stop();

    r1.restart("bench comm", Range::magenta);

    tm_total.start("bench-comm");

    for(IdxT cycle = 0; cycle < ncycles; cycle++) {

      Range r2("cycle", Range::yellow);

      IdxT imin = info.min[0];
      IdxT jmin = info.min[1];
      IdxT kmin = info.min[2];
      IdxT imax = info.max[0];
      IdxT jmax = info.max[1];
      IdxT kmax = info.max[2];
      IdxT ilen = info.len[0];
      IdxT jlen = info.len[1];
      IdxT klen = info.len[2];
      IdxT ijlen = info.stride[2];


      Range r3("pre-comm", Range::red);
      tm.start("pre-comm");

      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();

        for_all_3d(pol_loop{}, kmin, kmax,
                               jmin, jmax,
                               imin, imax,
                               detail::set_1(ilen, ijlen, data));
      }

      synchronize(pol_loop{});

      tm.stop();
      r3.restart("post-recv", Range::pink);
      tm.start("post-recv");

      comm.postRecv();

      tm.stop();
      r3.restart("post-send", Range::pink);
      tm.start("post-send");

      comm.postSend();

      tm.stop();
      r3.stop();

      /*
      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();

        for_all_3d(pol_loop{}, 0, klen,
                               0, jlen,
                               0, ilen,
                               [=] COMB_HOST COMB_DEVICE (IdxT k, IdxT j, IdxT i, IdxT idx) {
          IdxT zone = i + j * ilen + k * ijlen;
          DataT expected, found, next;
          if (k >= kmin && k < kmax &&
              j >= jmin && j < jmax &&
              i >= imin && i < imax) {
            expected = 1.0; found = data[zone]; next = 1.0;
          } else {
            expected = -1.0; found = data[zone]; next = -1.0;
          }
          // if (found != expected) {
          //   FPRINTF(stdout, "zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
          // }
          //FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
          data[zone] = next;
        });
      }
      */

      r3.start("wait-recv", Range::pink);
      tm.start("wait-recv");

      comm.waitRecv();

      tm.stop();
      r3.restart("wait-send", Range::pink);
      tm.start("wait-send");

      comm.waitSend();

      tm.stop();
      r3.restart("post-comm", Range::red);
      tm.start("post-comm");

      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();

        for_all_3d(pol_loop{}, 0, klen,
                               0, jlen,
                               0, ilen,
                               detail::reset_1(ilen, ijlen, data, imin, jmin, kmin, imax, jmax, kmax));
      }

      synchronize(pol_loop{});

      tm.stop();
      r3.stop();

      r2.stop();

    }

    comm.comminfo.barrier();

    tm_total.stop();

    r1.stop();

    print_timer(comm.comminfo, tm);
    print_timer(comm.comminfo, tm_total);

    tm.clear();
    tm_total.clear();
}

template < typename pol >
void do_warmup(COMB::Allocator& aloc, Timer& tm,  IdxT num_vars, IdxT len)
{
  tm.clear();

  char test_name[1024] = ""; snprintf(test_name, 1024, "warmup %s %s", pol::get_name(), aloc.name());
  Range r(test_name, Range::green);

  DataT** vars = new DataT*[num_vars];

  for (IdxT i = 0; i < num_vars; ++i) {
    vars[i] = (DataT*)aloc.allocate(len*sizeof(DataT));
  }

  for (IdxT i = 0; i < num_vars; ++i) {

    DataT* data = vars[i];

    for_all(pol{}, 0, len, detail::set_n1{data});
  }

  synchronize(pol{});

}

template < typename pol >
void do_copy(CommInfo& comminfo, COMB::Allocator& src_aloc, COMB::Allocator& dst_aloc, Timer& tm, IdxT num_vars, IdxT len, IdxT nrepeats)
{
  tm.clear();

  char test_name[1024] = ""; snprintf(test_name, 1024, "memcpy %s dst %s src %s", pol::get_name(), dst_aloc.name(), src_aloc.name());
  comminfo.print(FileGroup::all, "Starting test %s\n", test_name);

  Range r(test_name, Range::green);

  DataT** src = new DataT*[num_vars];
  DataT** dst = new DataT*[num_vars];

  for (IdxT i = 0; i < num_vars; ++i) {
    src[i] = (DataT*)src_aloc.allocate(len*sizeof(DataT));
    dst[i] = (DataT*)dst_aloc.allocate(len*sizeof(DataT));
  }

  // setup
  for (IdxT i = 0; i < num_vars; ++i) {
    for_all(pol{}, 0, len, detail::set_n1{dst[i]});
    for_all(pol{}, 0, len, detail::set_0{src[i]});
    for_all(pol{}, 0, len, detail::set_copy{dst[i], src[i]});
  }

  synchronize(pol{});

  char sub_test_name[1024] = ""; snprintf(sub_test_name, 1024, "copy_sync-%d-%d-%zu", num_vars, len, sizeof(DataT));

  for (IdxT rep = 0; rep < nrepeats; ++rep) {

    for (IdxT i = 0; i < num_vars; ++i) {
      for_all(pol{}, 0, len, detail::set_copy{src[i], dst[i]});
    }

    synchronize(pol{});

    tm.start(sub_test_name);

    for (IdxT i = 0; i < num_vars; ++i) {
      for_all(pol{}, 0, len, detail::set_copy{dst[i], src[i]});
    }

    synchronize(pol{});

    tm.stop();
  }

  print_timer(comminfo, tm);
  tm.clear();

  for (IdxT i = 0; i < num_vars; ++i) {
    dst_aloc.deallocate(dst[i]);
    src_aloc.deallocate(src[i]);
  }

  delete[] dst;
  delete[] src;
}

int main(int argc, char** argv)
{
  int required = MPI_THREAD_FUNNELED; // MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE
  int provided = detail::MPI::Init_thread(&argc, &argv, required);

  comb_setup_files(detail::MPI::Comm_rank(MPI_COMM_WORLD));

  { // begin region MPI communication via comminfo
  CommInfo comminfo;

  if (required != provided) {
    comminfo.print(FileGroup::err_master, "Didn't receive MPI thread support required %i provided %i.\n", required, provided);
    comminfo.abort();
  }

  comminfo.print(FileGroup::all, "Started rank %i of %i\n", comminfo.rank, comminfo.size);

  {
    char host[256];
    gethostname(host, 256);

    comminfo.print(FileGroup::all, "Node %s\n", host);
  }

  comminfo.print(FileGroup::all, "Compiler %s\n", COMB_SERIALIZE(COMB_COMPILER));

#ifdef COMB_ENABLE_CUDA
  comminfo.print(FileGroup::all, "Cuda compiler %s\n", COMB_SERIALIZE(COMB_CUDA_COMPILER));

  {
    const char* visible_devices = nullptr;
    visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    if (visible_devices == nullptr) {
      visible_devices = "undefined";
    }

    int device = -1;
    cudaCheck(cudaGetDevice(&device));

    comminfo.print(FileGroup::all, "GPU %i visible %s\n", device, visible_devices);
  }

  cudaCheck(cudaDeviceSynchronize());
#endif


  // read command line arguments
#ifdef COMB_ENABLE_OPENMP
  int omp_threads = -1;
#endif

  IdxT sizes[3] = {0, 0, 0};
  int divisions[3] = {0, 0, 0};
  int periodic[3] = {0, 0, 0};
  IdxT ghost_widths[3] = {1, 1, 1};
  IdxT num_vars = 1;
  IdxT ncycles = 5;

  // stores whether each exec policy is available for use
  ExecutorsAvailable exec_avail;
  exec_avail.seq = true;

  // stores whether each memory type is available for use
  COMB::AllocatorsAvailable memory_avail;
  memory_avail.host = true;

  IdxT i = 1;
  IdxT s = 0;
  for(; i < argc; ++i) {
    if (argv[i][0] == '-') {
      // options
      if (strcmp(&argv[i][1], "comm") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          ++i;
          if (strcmp(argv[i], "mock") == 0) {
            comminfo.mock_communication = true;
          } else if (strcmp(argv[i], "cutoff") == 0) {
            if (i+1 < argc && argv[i+1][0] != '-') {
              long read_cutoff = comminfo.cutoff;
              int ret = sscanf(argv[++i], "%ld", &read_cutoff);
              if (ret == 1) {
                comminfo.cutoff = read_cutoff;
              } else {
                comminfo.print(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              comminfo.print(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
            }
          } else if ( strcmp(argv[i], "post_recv") == 0
                   || strcmp(argv[i], "post_send") == 0
                   || strcmp(argv[i], "wait_recv") == 0
                   || strcmp(argv[i], "wait_send") == 0 ) {
            CommInfo::method* method = nullptr;
            if (strcmp(argv[i], "post_recv") == 0) {
              method = &comminfo.post_recv_method;
            } else if (strcmp(argv[i], "post_send") == 0) {
              method = &comminfo.post_send_method;
            } else if (strcmp(argv[i], "wait_recv") == 0) {
              method = &comminfo.wait_recv_method;
            } else if (strcmp(argv[i], "wait_send") == 0) {
              method = &comminfo.wait_send_method;
            }
            if (i+1 < argc && method != nullptr) {
              ++i;
              if (strcmp(argv[i], "wait_any") == 0) {
                *method = CommInfo::method::waitany;
              } else if (strcmp(argv[i], "wait_some") == 0) {
                *method = CommInfo::method::waitsome;
              } else if (strcmp(argv[i], "wait_all") == 0) {
                *method = CommInfo::method::waitall;
              } else if (strcmp(argv[i], "test_any") == 0) {
                *method = CommInfo::method::testany;
              } else if (strcmp(argv[i], "test_some") == 0) {
                *method = CommInfo::method::testsome;
              } else if (strcmp(argv[i], "test_all") == 0) {
                *method = CommInfo::method::testall;
              } else {
                comminfo.print(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              comminfo.print(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
            }
          } else {
            comminfo.print(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          comminfo.print(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "ghost") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          long read_ghost_widths[3] {ghost_widths[0], ghost_widths[1], ghost_widths[2]};
          int ret = sscanf(argv[++i], "%ld_%ld_%ld", &read_ghost_widths[0], &read_ghost_widths[1], &read_ghost_widths[2]);
          if (ret == 1) {
            ghost_widths[0] = read_ghost_widths[0];
            ghost_widths[1] = read_ghost_widths[0];
            ghost_widths[2] = read_ghost_widths[0];
          } else if (ret == 3) {
            ghost_widths[0] = read_ghost_widths[0];
            ghost_widths[1] = read_ghost_widths[1];
            ghost_widths[2] = read_ghost_widths[2];
          } else {
            comminfo.print(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          comminfo.print(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "exec") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          ++i;
          if ( strcmp(argv[i], "enable") == 0
            || strcmp(argv[i], "disable") == 0 ) {
            bool enabledisable = false;
            if (strcmp(argv[i], "enable") == 0) {
              enabledisable = true;
            } else if (strcmp(argv[i], "disable") == 0) {
              enabledisable = false;
            }
            if (i+1 < argc && argv[i+1][0] != '-') {
              ++i;
              if (strcmp(argv[i], "all") == 0) {
                exec_avail.seq = enabledisable;
  #ifdef COMB_ENABLE_OPENMP
                exec_avail.omp = enabledisable;
  #endif
  #ifdef COMB_ENABLE_CUDA
                exec_avail.cuda = enabledisable;
                exec_avail.cuda_batch = enabledisable;
                exec_avail.cuda_persistent = enabledisable;
                exec_avail.cuda_batch_fewgs = enabledisable;
                exec_avail.cuda_persistent_fewgs = enabledisable;
  #endif
  #ifdef COMB_ENABLE_CUDA_GRAPH
                exec_avail.cuda_graph = enabledisable;
  #endif
                exec_avail.mpi_type = enabledisable;
              } else if (strcmp(argv[i], "seq") == 0) {
                exec_avail.seq = enabledisable;
              } else if (strcmp(argv[i], "omp") == 0 ||
                         strcmp(argv[i], "openmp") == 0) {
  #ifdef COMB_ENABLE_OPENMP
                exec_avail.omp = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda") == 0) {
  #ifdef COMB_ENABLE_CUDA
                exec_avail.cuda = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_batch") == 0) {
  #ifdef COMB_ENABLE_CUDA
                exec_avail.cuda_batch = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_persistent") == 0) {
  #ifdef COMB_ENABLE_CUDA
                exec_avail.cuda_persistent = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_batch_fewgs") == 0) {
  #ifdef COMB_ENABLE_CUDA
                exec_avail.cuda_batch_fewgs = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_persistent_fewgs") == 0) {
  #ifdef COMB_ENABLE_CUDA
                exec_avail.cuda_persistent_fewgs = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_graph") == 0) {
  #ifdef COMB_ENABLE_CUDA_GRAPH
                exec_avail.cuda_graph = enabledisable;
  #endif
              } else if (strcmp(argv[i], "mpi_type") == 0) {
                exec_avail.mpi_type = enabledisable;
              } else {
                comminfo.print(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              comminfo.print(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
            }
          } else {
            comminfo.print(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          comminfo.print(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "memory") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          ++i;
          if ( strcmp(argv[i], "enable") == 0
            || strcmp(argv[i], "disable") == 0 ) {
            bool enabledisable = false;
            if (strcmp(argv[i], "enable") == 0) {
              enabledisable = true;
            } else if (strcmp(argv[i], "disable") == 0) {
              enabledisable = false;
            }
            if (i+1 < argc && argv[i+1][0] != '-') {
              ++i;
              if (strcmp(argv[i], "all") == 0) {
                memory_avail.host = enabledisable;
  #ifdef COMB_ENABLE_CUDA
                memory_avail.cuda_pinned = enabledisable;
                memory_avail.cuda_device = enabledisable;
                memory_avail.cuda_managed = enabledisable;
                memory_avail.cuda_managed_host_preferred = enabledisable;
                memory_avail.cuda_managed_host_preferred_device_accessed = enabledisable;
                memory_avail.cuda_managed_device_preferred = enabledisable;
                memory_avail.cuda_managed_device_preferred_host_accessed = enabledisable;
  #endif
              } else if (strcmp(argv[i], "host") == 0) {
                memory_avail.host = enabledisable;
              } else if (strcmp(argv[i], "cuda_pinned") == 0) {
  #ifdef COMB_ENABLE_CUDA
                memory_avail.cuda_pinned = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_device") == 0) {
  #ifdef COMB_ENABLE_CUDA
                memory_avail.cuda_device = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_managed") == 0) {
  #ifdef COMB_ENABLE_CUDA
                memory_avail.cuda_managed = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_managed_host_preferred") == 0) {
  #ifdef COMB_ENABLE_CUDA
                memory_avail.cuda_managed_host_preferred = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_managed_host_preferred_device_accessed") == 0) {
  #ifdef COMB_ENABLE_CUDA
                memory_avail.cuda_managed_host_preferred_device_accessed = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_managed_device_preferred") == 0) {
  #ifdef COMB_ENABLE_CUDA
                memory_avail.cuda_managed_device_preferred = enabledisable;
  #endif
              } else if (strcmp(argv[i], "cuda_managed_device_preferred_host_accessed") == 0) {
  #ifdef COMB_ENABLE_CUDA
                memory_avail.cuda_managed_device_preferred_host_accessed = enabledisable;
  #endif
              } else {
                comminfo.print(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              comminfo.print(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
            }
          } else {
            comminfo.print(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          comminfo.print(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "vars") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          long read_num_vars = num_vars;
          int ret = sscanf(argv[++i], "%ld", &read_num_vars);
          if (ret == 1) {
            num_vars = read_num_vars;
          } else {
            comminfo.print(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          comminfo.print(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "cycles") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          long read_ncycles = ncycles;
          int ret = sscanf(argv[++i], "%ld", &read_ncycles);
          if (ret == 1) {
            ncycles = read_ncycles;
          } else {
            comminfo.print(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          comminfo.print(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "periodic") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          long read_periodic[3] {periodic[0], periodic[1], periodic[2]};
          int ret = sscanf(argv[++i], "%ld_%ld_%ld", &read_periodic[0], &read_periodic[1], &read_periodic[2]);
          if (ret == 1) {
            periodic[0] = read_periodic[0] ? 1 : 0;
            periodic[1] = read_periodic[0] ? 1 : 0;
            periodic[2] = read_periodic[0] ? 1 : 0;
          } else if (ret == 3) {
            periodic[0] = read_periodic[0] ? 1 : 0;
            periodic[1] = read_periodic[1] ? 1 : 0;
            periodic[2] = read_periodic[2] ? 1 : 0;
          } else {
            comminfo.print(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          comminfo.print(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "divide") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          long read_divisions[3] {divisions[0], divisions[1], divisions[2]};
          int ret = sscanf(argv[++i], "%ld_%ld_%ld", &read_divisions[0], &read_divisions[1], &read_divisions[2]);
          if (ret == 1) {
            divisions[0] = read_divisions[0];
            divisions[1] = read_divisions[0];
            divisions[2] = read_divisions[0];
          } else if (ret == 3) {
            divisions[0] = read_divisions[0];
            divisions[1] = read_divisions[1];
            divisions[2] = read_divisions[2];
          } else {
            comminfo.print(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          comminfo.print(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "omp_threads") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
#ifdef COMB_ENABLE_OPENMP
          long read_omp_threads = omp_threads;
#else
          long read_omp_threads = 0;
#endif
          int ret = sscanf(argv[++i], "%ld", &read_omp_threads);
          if (ret == 1) {
#ifdef COMB_ENABLE_OPENMP
            omp_threads = read_omp_threads;
#else
            comminfo.print(FileGroup::err_master, "Not built with openmp, ignoring %s %s.\n", argv[i-1], argv[i]);
#endif
          } else {
            comminfo.print(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          comminfo.print(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "cuda_aware_mpi") == 0) {
#ifdef COMB_ENABLE_CUDA
        exec_avail.cuda_aware_mpi = true;
#else
        comminfo.print(FileGroup::err_master, "Not built with cuda, ignoring %s.\n", argv[i]);
#endif
      } else if (strcmp(&argv[i][1], "cuda_host_accessible_from_device") == 0) {
#ifdef COMB_ENABLE_CUDA
        memory_avail.cuda_host_accessible_from_device = detail::cuda::get_host_accessible_from_device();
#else
        comminfo.print(FileGroup::err_master, "Not built with cuda, ignoring %s.\n", argv[i]);
#endif
      } else {
        comminfo.print(FileGroup::err_master, "Unknown option, ignoring %s.\n", argv[i]);
      }
    } else if (std::isdigit(argv[i][0]) && s < 1) {
      long read_sizes[3] {sizes[0], sizes[1], sizes[2]};
      int ret = sscanf(argv[i], "%ld_%ld_%ld", &read_sizes[0], &read_sizes[1], &read_sizes[2]);
      if (ret == 1) {
        ++s;
        sizes[0] = read_sizes[0];
        sizes[1] = read_sizes[0];
        sizes[2] = read_sizes[0];
      } else if (ret == 3) {
        ++s;
        sizes[0] = read_sizes[0];
        sizes[1] = read_sizes[1];
        sizes[2] = read_sizes[2];
      } else {
        comminfo.print(FileGroup::err_master, "Invalid argument to sizes, ignoring %s.\n", argv[i]);
      }
    } else {
      comminfo.print(FileGroup::err_master, "Invalid argument, ignoring %s.\n", argv[i]);
    }
  }

  if (ncycles <= 0) {
    comminfo.print(FileGroup::err_master, "Invalid cycles argument.\n");
    comminfo.abort();
  } else if (num_vars <= 0) {
    comminfo.print(FileGroup::err_master, "Invalid vars argument.\n");
    comminfo.abort();
  } else if ( (ghost_widths[0] <  0 || ghost_widths[1] <  0 || ghost_widths[2] <  0)
           || (ghost_widths[0] == 0 && ghost_widths[1] == 0 && ghost_widths[2] == 0) ) {
    comminfo.print(FileGroup::err_master, "Invalid ghost widths.\n");
    comminfo.abort();
  } else if ( (divisions[0] != 0 || divisions[1] != 0 || divisions[2] != 0)
           && (comminfo.size != divisions[0] * divisions[1] * divisions[2]) ) {
    comminfo.print(FileGroup::err_master, "Invalid mesh divisions\n");
    comminfo.abort();
  }

#ifdef COMB_ENABLE_OPENMP
  // OMP setup
  {
    if (omp_threads > 0) {

      omp_set_num_threads(omp_threads);

    }

#pragma omp parallel shared(omp_threads)
    {
#pragma omp master
      omp_threads = omp_get_num_threads();
    }

    long print_omp_threads = omp_threads;
    comminfo.print(FileGroup::all, "OMP num threads %5li\n", print_omp_threads);

#ifdef PRINT_THREAD_MAP
    {
      int* thread_cpu_id = new int[omp_threads];

#pragma omp parallel shared(thread_cpu_id)
      {
        int thread_id = omp_get_thread_num();

        thread_cpu_id[thread_id] = sched_getcpu();
      }

      int i = 0;
      if (i < omp_threads) {
        comminfo.print(FileGroup::all, "OMP thread map %6i", thread_cpu_id[i]);
        for (++i; i < omp_threads; ++i) {
          comminfo.print(FileGroup::all, " %8i", thread_cpu_id[i]);
        }
      }

      comminfo.print(FileGroup::all, "\n");

      delete[] thread_cpu_id;

    }
#endif // ifdef PRINT_THREAD_MAP
  }
#endif // ifdef COMB_ENABLE_OPENMP


  GlobalMeshInfo global_info(sizes, comminfo.size, divisions, periodic, ghost_widths);

  // create cartesian communicator and get rank
  comminfo.cart.create(global_info.divisions, global_info.periodic);

  MeshInfo info = MeshInfo::get_local(global_info, comminfo.cart.coords);

  // print info about problem setup
  {
    long print_coords[3]       = {comminfo.cart.coords[0],    comminfo.cart.coords[1],    comminfo.cart.coords[2]   };
    long print_cutoff          = comminfo.cutoff;
    long print_ncycles         = ncycles;
    long print_num_vars        = num_vars;
    long print_ghost_widths[3] = {info.ghost_widths[0],       info.ghost_widths[1],       info.ghost_widths[2]      };
    long print_sizes[3]        = {global_info.sizes[0],       global_info.sizes[1],       global_info.sizes[2]      };
    long print_divisions[3]    = {comminfo.cart.divisions[0], comminfo.cart.divisions[1], comminfo.cart.divisions[2]};
    long print_periodic[3]     = {comminfo.cart.periodic[0],  comminfo.cart.periodic[1],  comminfo.cart.periodic[2] };

    comminfo.print(FileGroup::all, "Do %s communication\n",         comminfo.mock_communication ? "mock" : "real"                      );
    comminfo.print(FileGroup::all, "Cart coords  %8li %8li %8li\n", print_coords[0],       print_coords[1],       print_coords[2]      );
    comminfo.print(FileGroup::all, "Message policy cutoff %li\n",   print_cutoff                                                       );
    comminfo.print(FileGroup::all, "Post Recv using %s method\n",   CommInfo::method_str(comminfo.post_recv_method)                    );
    comminfo.print(FileGroup::all, "Post Send using %s method\n",   CommInfo::method_str(comminfo.post_send_method)                    );
    comminfo.print(FileGroup::all, "Wait Recv using %s method\n",   CommInfo::method_str(comminfo.wait_recv_method)                    );
    comminfo.print(FileGroup::all, "Wait Send using %s method\n",   CommInfo::method_str(comminfo.wait_send_method)                    );
    comminfo.print(FileGroup::all, "Num cycles   %8li\n",           print_ncycles                                                      );
    comminfo.print(FileGroup::all, "Num vars     %8li\n",           print_num_vars                                                     );
    comminfo.print(FileGroup::all, "ghost_widths %8li %8li %8li\n", print_ghost_widths[0], print_ghost_widths[1], print_ghost_widths[2]);
    comminfo.print(FileGroup::all, "sizes        %8li %8li %8li\n", print_sizes[0],        print_sizes[1],        print_sizes[2]       );
    comminfo.print(FileGroup::all, "divisions    %8li %8li %8li\n", print_divisions[0],    print_divisions[1],    print_divisions[2]   );
    comminfo.print(FileGroup::all, "periodic     %8li %8li %8li\n", print_periodic[0],     print_periodic[1],     print_periodic[2]    );
    comminfo.print(FileGroup::all, "division map\n");
    // print division map
    IdxT max_cuts = std::max(std::max(comminfo.cart.divisions[0], comminfo.cart.divisions[1]), comminfo.cart.divisions[2]);
    for (IdxT ci = 0; ci <= max_cuts; ++ci) {
      comminfo.print(FileGroup::all, "map         ");
      if (ci <= comminfo.cart.divisions[0]) {
        long print_division_coord = ci * (sizes[0] / comminfo.cart.divisions[0]) + std::min(ci, sizes[0] % comminfo.cart.divisions[0]);
        comminfo.print(FileGroup::all, " %8li", print_division_coord);
      } else {
        comminfo.print(FileGroup::all, " %8s", "");
      }
      if (ci <= comminfo.cart.divisions[1]) {
        long print_division_coord = ci * (sizes[1] / comminfo.cart.divisions[1]) + std::min(ci, sizes[1] % comminfo.cart.divisions[1]);
        comminfo.print(FileGroup::all, " %8li", print_division_coord);
      } else {
        comminfo.print(FileGroup::all, " %8s", "");
      }
      if (ci <= comminfo.cart.divisions[2]) {
        long print_division_coord = ci * (sizes[2] / comminfo.cart.divisions[2]) + std::min(ci, sizes[2] % comminfo.cart.divisions[2]);
        comminfo.print(FileGroup::all, " %8li", print_division_coord);
      } else {
        comminfo.print(FileGroup::all, " %8s", "");
      }
      comminfo.print(FileGroup::all, "\n");
    }
  }

  COMB::Allocators alloc;

  Timer tm(2*6*ncycles);
  Timer tm_total(1024);

  // warm-up memory pools
  {
    do_warmup<seq_pol>(alloc.host, tm, num_vars+1, info.totallen);

#ifdef COMB_ENABLE_OPENMP
    do_warmup<omp_pol>(alloc.host, tm, num_vars+1, info.totallen);
#endif

#ifdef COMB_ENABLE_CUDA
    do_warmup<seq_pol>(alloc.hostpinned, tm, num_vars+1, info.totallen);

    do_warmup<cuda_pol>(alloc.device, tm, num_vars+1, info.totallen);

    do_warmup<seq_pol>( alloc.managed, tm, num_vars+1, info.totallen);
    do_warmup<cuda_pol>(alloc.managed, tm, num_vars+1, info.totallen);

    do_warmup<seq_pol>(       alloc.managed_host_preferred, tm, num_vars+1, info.totallen);
    do_warmup<cuda_batch_pol>(alloc.managed_host_preferred, tm, num_vars+1, info.totallen);

    do_warmup<seq_pol>(            alloc.managed_host_preferred_device_accessed, tm, num_vars+1, info.totallen);
    do_warmup<cuda_persistent_pol>(alloc.managed_host_preferred_device_accessed, tm, num_vars+1, info.totallen);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      do_warmup<seq_pol>(       alloc.managed_device_preferred, tm, num_vars+1, info.totallen);
      do_warmup<cuda_batch_pol>(alloc.managed_device_preferred, tm, num_vars+1, info.totallen);

      do_warmup<seq_pol>(            alloc.managed_device_preferred_host_accessed, tm, num_vars+1, info.totallen);
      do_warmup<cuda_persistent_pol>(alloc.managed_device_preferred_host_accessed, tm, num_vars+1, info.totallen);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    do_warmup<cuda_graph_pol>(alloc.device, tm, num_vars+1, info.totallen);
#endif
#endif

  }

  // host memory
  if (memory_avail.host) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.host.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, info.totallen, ncycles);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, info.totallen, ncycles);
#endif

#ifdef COMB_ENABLE_CUDA
    if (memory_avail.cuda_host_accessible_from_device) {

      if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, info.totallen, ncycles);

      if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, info.totallen, ncycles);

      if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, info.totallen, ncycles);

      {
        SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

        if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, info.totallen, ncycles);

        if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, info.totallen, ncycles);
      }

#ifdef COMB_ENABLE_CUDA_GRAPH
      if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.host, alloc.host, tm, num_vars, info.totallen, ncycles);
#endif
    }
#endif
  }

#ifdef COMB_ENABLE_CUDA
  if (memory_avail.cuda_pinned) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.hostpinned.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.hostpinned, alloc.host, tm, num_vars, info.totallen, ncycles);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.hostpinned, alloc.host, tm, num_vars, info.totallen, ncycles);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.hostpinned, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
#endif
  }

  if (memory_avail.cuda_device) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.device.name());
    Range r0(name, Range::green);

    if (detail::cuda::get_device_accessible_from_host()) {
      if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.device, alloc.host, tm, num_vars, info.totallen, ncycles);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.device, alloc.host, tm, num_vars, info.totallen, ncycles);
#endif
    }

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.device, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
#endif
  }

  if (memory_avail.cuda_managed) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed, alloc.host, tm, num_vars, info.totallen, ncycles);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed, alloc.host, tm, num_vars, info.totallen, ncycles);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
#endif
  }

  if (memory_avail.cuda_managed_host_preferred) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed_host_preferred.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed_host_preferred, alloc.host, tm, num_vars, info.totallen, ncycles);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed_host_preferred, alloc.host, tm, num_vars, info.totallen, ncycles);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed_host_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
#endif
  }

  if (memory_avail.cuda_managed_host_preferred_device_accessed) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed_host_preferred_device_accessed.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.host, tm, num_vars, info.totallen, ncycles);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.host, tm, num_vars, info.totallen, ncycles);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed_host_preferred_device_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
#endif
  }

  if (memory_avail.cuda_managed_device_preferred) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed_device_preferred.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed_device_preferred, alloc.host, tm, num_vars, info.totallen, ncycles);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed_device_preferred, alloc.host, tm, num_vars, info.totallen, ncycles);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed_device_preferred, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
#endif
  }

  if (memory_avail.cuda_managed_device_preferred_host_accessed) {
    char name[1024] = ""; snprintf(name, 1024, "set_vars %s", alloc.managed_device_preferred_host_accessed.name());
    Range r0(name, Range::green);

    if (exec_avail.seq) do_copy<seq_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.host, tm, num_vars, info.totallen, ncycles);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp) do_copy<omp_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.host, tm, num_vars, info.totallen, ncycles);
#endif

    if (exec_avail.cuda) do_copy<cuda_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_batch) do_copy<cuda_batch_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    if (exec_avail.cuda_persistent) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

    {
      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda_batch_fewgs) do_copy<cuda_batch_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);

      if (exec_avail.cuda_persistent_fewgs) do_copy<cuda_persistent_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda_graph) do_copy<cuda_graph_pol>(comminfo, alloc.managed_device_preferred_host_accessed, alloc.hostpinned, tm, num_vars, info.totallen, ncycles);
#endif
  }
#endif // COMB_ENABLE_CUDA

  // host allocated
  if (memory_avail.host) {
    COMB::Allocator& mesh_aloc = alloc.host;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
    if (memory_avail.cuda_host_accessible_from_device) {
      if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
        do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
        do_cycles<cuda_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
        do_cycles<cuda_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
        do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
        do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      {
        if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
          do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
          do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
          do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
          do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);


        SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

        if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
          do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
          do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
          do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

        if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
          do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
      }

#ifdef COMB_ENABLE_CUDA_GRAPH
      if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
        do_cycles<cuda_pol, cuda_graph_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
        do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif
    }
#endif

    if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<seq_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<omp_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
    if (memory_avail.cuda_host_accessible_from_device) {
      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
#endif
  }

#ifdef COMB_ENABLE_CUDA
  // host pinned allocated
  if (memory_avail.cuda_pinned) {
    COMB::Allocator& mesh_aloc = alloc.hostpinned;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<seq_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<omp_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
      do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
  }

  // device allocated
  if (memory_avail.cuda_device) {
    COMB::Allocator& mesh_aloc = alloc.device;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (detail::cuda::get_device_accessible_from_host()) {
      if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
        do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
        do_cycles<omp_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
        do_cycles<omp_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
        do_cycles<omp_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
        do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
        do_cycles<cuda_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
        do_cycles<cuda_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
        do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
    }

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
          do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
          do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
          do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
          do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
      }

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (detail::cuda::get_device_accessible_from_host()) {
      if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
        do_cycles<cuda_pol, cuda_graph_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);
    }

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
          do_cycles<seq_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
      }

#ifdef COMB_ENABLE_OPENMP
      if (detail::cuda::get_device_accessible_from_host()) {
        if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
          do_cycles<omp_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
      }
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed allocated
  if (memory_avail.cuda_managed) {
    COMB::Allocator& mesh_aloc = alloc.managed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed host preferred allocated
  if (memory_avail.cuda_managed_host_preferred) {
    COMB::Allocator& mesh_aloc = alloc.managed_host_preferred;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed host preferred device accessed allocated
  if (memory_avail.cuda_managed_host_preferred_device_accessed) {
    COMB::Allocator& mesh_aloc = alloc.managed_host_preferred_device_accessed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed device preferred allocated
  if (memory_avail.cuda_managed_device_preferred) {
    COMB::Allocator& mesh_aloc = alloc.managed_device_preferred;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }

  // managed device preferred host accessed allocated
  if (memory_avail.cuda_managed_device_preferred_host_accessed) {
    COMB::Allocator& mesh_aloc = alloc.managed_device_preferred_host_accessed;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    if (exec_avail.seq && exec_avail.seq && exec_avail.seq)
      do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.omp && exec_avail.seq && exec_avail.seq)
      do_cycles<omp_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.seq)
      do_cycles<omp_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.omp && exec_avail.omp && exec_avail.omp)
      do_cycles<omp_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.seq && exec_avail.seq)
      do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
    if (exec_avail.cuda && exec_avail.omp && exec_avail.seq)
      do_cycles<cuda_pol, omp_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.omp && exec_avail.omp)
      do_cycles<cuda_pol, omp_pol, omp_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.host, alloc.host, tm, tm_total);
#endif

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq)
      do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda)
      do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

    {
      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);


      SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs)
        do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq)
        do_cycles<cuda_pol, cuda_persistent_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

      if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs)
        do_cycles<cuda_pol, cuda_persistent_pol, cuda_persistent_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
    }

#ifdef COMB_ENABLE_CUDA_GRAPH
    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq)
      do_cycles<cuda_pol, cuda_graph_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.host, tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph)
      do_cycles<cuda_pol, cuda_graph_pol, cuda_graph_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, alloc.hostpinned, alloc.hostpinned, tm, tm_total);
#endif

    if (exec_avail.cuda_aware_mpi) {
      if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<seq_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
      if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<omp_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
#endif

      if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type)
        do_cycles<cuda_pol, mpi_type_pol, mpi_type_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, mesh_aloc, mesh_aloc, tm, tm_total);
    }
  }
#endif // COMB_ENABLE_CUDA

  } // end region MPI communication via comminfo

  comb_teardown_files();

  detail::MPI::Finalize();
  return 0;
}

