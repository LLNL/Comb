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

#ifndef _DO_CYCLES_HPP
#define _DO_CYCLES_HPP

#include "comb.hpp"

namespace COMB {

template < typename pol_loop, typename pol_many, typename pol_few >
void do_cycles(CommInfo& comm_info, MeshInfo& info,
               IdxT num_vars, IdxT ncycles,
               COMB::Allocator& aloc_mesh,
               COMB::Allocator& aloc_many, COMB::Allocator& aloc_few,
               Timer& tm, Timer& tm_total)
{
    tm_total.clear();
    tm.clear();

    char rname[1024] = ""; snprintf(rname, 1024, "Buffers %s %s %s %s", pol_many::get_name(), aloc_many.name(), pol_few::get_name(), aloc_few.name());
    char test_name[1024] = ""; snprintf(test_name, 1024, "Mesh %s %s %s %s", pol_loop::get_name(), aloc_mesh.name(), rname);
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

      bool mock_communication = comm.mock_communication();
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

} // namespace COMB

#endif // _DO_CYCLES_HPP

