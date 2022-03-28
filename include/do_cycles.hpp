//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2022, Lawrence Livermore National Security, LLC.
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

#include <type_traits>

#include "comb.hpp"
#include "CommFactory.hpp"

namespace COMB {

template < typename pol_comm, typename exec_mesh, typename exec_many, typename exec_few >
bool should_do_cycles(CommContext<pol_comm>& con_comm_in,
                      ContextHolder<exec_mesh>& con_mesh_in, AllocatorInfo& aloc_mesh_in,
                      ContextHolder<exec_many>& con_many_in, AllocatorInfo& aloc_many_in,
                      ContextHolder<exec_few>& con_few_in,   AllocatorInfo& aloc_few_in)
{
  return con_mesh_in.available() && con_many_in.available() && con_few_in.available()
      && aloc_mesh_in.available(AllocatorInfo::UseType::Mesh)
      && aloc_many_in.available(AllocatorInfo::UseType::Buffer)
      && aloc_few_in.available(AllocatorInfo::UseType::Buffer)
      && aloc_many_in.accessible(con_comm_in) && aloc_few_in.accessible(con_comm_in)
      && aloc_mesh_in.accessible(con_mesh_in.get())
      && aloc_mesh_in.accessible(con_many_in.get()) && aloc_many_in.accessible(con_many_in.get())
      && aloc_mesh_in.accessible(con_few_in.get())  && aloc_few_in.accessible(con_few_in.get()) ;
}

template < typename pol_comm, typename exec_mesh, typename exec_many, typename exec_few >
void do_cycles(CommContext<pol_comm>& con_comm_in,
               CommInfo& comm_info, MeshInfo& info,
               IdxT num_vars, IdxT ncycles,
               ContextHolder<exec_mesh>& con_mesh_in, AllocatorInfo& aloc_mesh_in,
               ContextHolder<exec_many>& con_many_in, AllocatorInfo& aloc_many_in,
               ContextHolder<exec_few>& con_few_in,   AllocatorInfo& aloc_few_in,
               Timer& tm, Timer& tm_total)
{
  if (!should_do_cycles(con_comm_in, con_mesh_in, aloc_mesh_in, con_many_in, aloc_many_in, con_few_in, aloc_few_in)) {
    return;
  }

  using con_mesh_type = typename ContextHolder<exec_mesh>::context_type;
  using con_many_type = typename ContextHolder<exec_many>::context_type;
  using con_few_type  = typename ContextHolder<exec_few>::context_type;
  using pol_mesh = typename con_mesh_type::pol;
  using pol_many = typename con_many_type::pol;
  using pol_few  = typename con_few_type::pol;

  ExecContext<pol_mesh>& con_mesh = con_mesh_in.get();
  ExecContext<pol_many>& con_many = con_many_in.get();
  ExecContext<pol_few>&  con_few  = con_few_in.get();

  COMB::Allocator& aloc_mesh = aloc_mesh_in.allocator();
  COMB::Allocator& aloc_many = aloc_many_in.allocator();
  COMB::Allocator& aloc_few  = aloc_few_in.allocator();


  CPUContext tm_con;
  tm_total.clear();
  tm.clear();

  char test_name[1024] = ""; snprintf(test_name, 1024, "Comm %s Mesh %s %s Buffers %s %s %s %s",
                                                        pol_comm::get_name(),
                                                        pol_mesh::get_name(), aloc_mesh.name(),
                                                        pol_many::get_name(), aloc_many.name(), pol_few::get_name(), aloc_few.name());
  fgprintf(FileGroup::all, "Starting test %s\n", test_name);

  {
    Range r0(test_name, Range::orange);

    // make a copy of comminfo to duplicate the MPI communicator
    CommInfo comminfo(comm_info);

    using comm_type = Comm<pol_many, pol_few, pol_comm>;

#ifdef COMB_ENABLE_MPI
    // set name of communicator
    // include name of memory space if using mpi datatypes for pack/unpack
    char comm_name[MPI_MAX_OBJECT_NAME] = "";
    snprintf(comm_name, MPI_MAX_OBJECT_NAME, "COMB_MPI_CART_COMM%s%s",
        (comm_type::use_mpi_type) ? "_"              : "",
        (comm_type::use_mpi_type) ? aloc_mesh.name() : "");

    comminfo.set_name(comm_name);
#endif

    CommContext<pol_comm> con_comm(con_comm_in
#ifdef COMB_ENABLE_MPI
                                  ,comminfo.cart.comm
#endif
                                   );

    // sometimes set cutoff to 0 (always use pol_many) to simplify algorithms
    if (std::is_same<pol_many, pol_few>::value) {
      // check comm send (packing) method
      switch (comminfo.post_send_method) {
        case CommInfo::method::waitsome:
        case CommInfo::method::testsome:
          // don't change cutoff to see if breaking messages into
          // sized groups matters
          break;
        default:
          // already packing individually or all together
          // might aw well use simpler algorithm
          comminfo.cutoff = 0;
          break;
      }
    }

    // make communicator object
    comm_type comm(con_comm, comminfo, aloc_mesh, aloc_many, aloc_few);

    comm.barrier();

    tm_total.start(tm_con, "start-up");

    std::vector<MeshData> vars;
    vars.reserve(num_vars);

    {
      Range r2("setup factory", Range::yellow);

      CommFactory factory(comminfo);

      r2.restart("add vars", Range::yellow);

      for (IdxT i = 0; i < num_vars; ++i) {

        vars.push_back(MeshData(info, aloc_mesh));

        vars[i].allocate();

        DataT* data = vars[i].data();
        IdxT totallen = info.totallen;

        con_mesh.for_all(totallen,
                         [=] COMB_HOST COMB_DEVICE (IdxT i) {
          // LOGPRINTF("init-var %p[%i] = %f\n", data, i, -1.0);
          data[i] = -1.0;
        });

        factory.add_var(vars[i]);

        con_mesh.synchronize();
      }

      r2.restart("populate comm", Range::yellow);

      factory.populate(comm, con_many, con_few);
    }

    tm_total.stop(tm_con);

    comm.barrier();

    Range r1("test correctness", Range::indigo);

    tm_total.start(tm_con, "test-comm");

    if (comm_type::persistent) {
      // tm.start(tm_con, "init-persistent-comm");
      comm.init_persistent_comm(con_many, con_few);
      // tm.stop(tm_con);
    }

    IdxT ntestcycles = std::max(IdxT{1}, ncycles/IdxT{10});
    for (IdxT test_cycle = 0; test_cycle < ntestcycles; ++test_cycle) { // test comm

      char cycle_range_name[32] = "";
      snprintf(cycle_range_name, 32, "cycle_%lli", (long long)test_cycle);
      Range r2(cycle_range_name, Range::cyan);

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


      Range r3("pre-comm", Range::red);
      // tm.start(tm_con, "pre-comm");

      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();
        IdxT var_i = i + 1;

        con_mesh.for_all_3d(klen,
                            jlen,
                            ilen,
                            [=] COMB_HOST COMB_DEVICE (IdxT k, IdxT j, IdxT i) {
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
              FGPRINTF(FileGroup::proc, "test pre-comm %p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
            }
            // LOGPRINTF("test pre-comm %p[%i]{%f} = %f\n", data, zone, found, next);
            assert(found == expected);
          }
          data[zone] = next;
        });
      }

      con_mesh.synchronize();

      // tm.stop(tm_con);
      r3.restart("post-recv", Range::pink);
      // tm.start(tm_con, "post-recv");

      comm.postRecv(con_many, con_few);

      // tm.stop(tm_con);
      r3.restart("post-send", Range::pink);
      // tm.start(tm_con, "post-send");

      comm.postSend(con_many, con_few);

      // tm.stop(tm_con);
      r3.stop();

      // for (IdxT i = 0; i < num_vars; ++i) {

      //   DataT* data = vars[i].data();
      //   IdxT var_i = i + 1;

      //   con_mesh.for_all_3d(klen,
      //                       jlen,
      //                       ilen,
      //                       [=] COMB_HOST COMB_DEVICE (IdxT k, IdxT j, IdxT i) {
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
      //           FGPRINTF(FileGroup::proc, "test mid-comm %p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
      //         }
      //         LOGPRINTF("test mid-comm %p[%i]{%f} = %f\n", data, zone, found, next);
      //         assert(found == expected);
      //       }
      //       data[zone] = next;
      //     }
      //     // other zones may be participating in communication, do not access
      //   });
      // }

      // con_mesh.synchronize();


      r3.start("wait-recv", Range::pink);
      // tm.start(tm_con, "wait-recv");

      comm.waitRecv(con_many, con_few);

      // tm.stop(tm_con);
      r3.restart("wait-send", Range::pink);
      // tm.start(tm_con, "wait-send");

      comm.waitSend(con_many, con_few);

      // tm.stop(tm_con);
      r3.restart("post-comm", Range::red);
      // tm.start(tm_con, "post-comm");

      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();
        IdxT var_i = i + 1;

        con_mesh.for_all_3d(klen,
                            jlen,
                            ilen,
                            [=] COMB_HOST COMB_DEVICE (IdxT k, IdxT j, IdxT i) {
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
            expected =-(zone_global+var_i); found = data[zone]; next = -1.0;
            branchid = 0;
          } else if (k >= kmin && k < kmax &&
                     j >= jmin && j < jmax &&
                     i >= imin && i < imax) {
            // interior communicated zones should not have changed value
            expected = zone_global + var_i; found = data[zone]; next = -1.0;
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
              FGPRINTF(FileGroup::proc, "test post-comm %p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
            }
            // LOGPRINTF("test post-comm %p[%i]{%p} = %f\n", data, zone, found, next);
            assert(found == expected);
          }
          data[zone] = next;
        });
      }

      con_mesh.synchronize();

      // tm.stop(tm_con);
      r3.stop();

      r2.stop();
    }

    if (comm_type::persistent) {
      // tm.start(tm_con, "cleanup-persistent-comm");
      comm.cleanup_persistent_comm();
      // tm.stop(tm_con);
    }

    comm.barrier();

    tm_total.stop(tm_con);

    tm.clear();

    r1.restart("bench comm", Range::magenta);

    tm_total.start(tm_con, "bench-comm");

    if (comm_type::persistent) {
      tm.start(tm_con, "init-persistent-comm");
      comm.init_persistent_comm(con_many, con_few);
      tm.stop(tm_con);
    }

    for(IdxT cycle = 0; cycle < ncycles; cycle++) {

      char cycle_range_name[32] = "";
      snprintf(cycle_range_name, 32, "cycle_%lli", (long long)cycle);
      Range r2(cycle_range_name, Range::yellow);

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
      tm.start(tm_con, "pre-comm");

      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();

        // set internal zones to 1
        con_mesh.for_all_3d(kmax - kmin,
                            jmax - jmin,
                            imax - imin,
                            detail::set_1(ilen, ijlen, data, imin, jmin, kmin));
      }

      con_mesh.synchronize();

      tm.stop(tm_con);
      r3.restart("post-recv", Range::pink);
      tm.start(tm_con, "post-recv");

      comm.postRecv(con_many, con_few);

      tm.stop(tm_con);
      r3.restart("post-send", Range::pink);
      tm.start(tm_con, "post-send");

      comm.postSend(con_many, con_few);

      tm.stop(tm_con);
      r3.stop();

      /*
      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();

        con_mesh.for_all_3d(klen,
                            jlen,
                            ilen,
                            [=] COMB_HOST COMB_DEVICE (IdxT k, IdxT j, IdxT i) {
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
          //   FGPRINTF(FileGroup::proc, "zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
          // }
          // LOGPRINTF("%p[%i] = %f\n", data, zone, 1.0);
          data[zone] = next;
        });
      }
      */

      r3.start("wait-recv", Range::pink);
      tm.start(tm_con, "wait-recv");

      comm.waitRecv(con_many, con_few);

      tm.stop(tm_con);
      r3.restart("wait-send", Range::pink);
      tm.start(tm_con, "wait-send");

      comm.waitSend(con_many, con_few);

      tm.stop(tm_con);
      r3.restart("post-comm", Range::red);
      tm.start(tm_con, "post-comm");

      for (IdxT i = 0; i < num_vars; ++i) {

        DataT* data = vars[i].data();

        // set all zones to 1
        con_mesh.for_all_3d(klen,
                            jlen,
                            ilen,
                            detail::set_1(ilen, ijlen, data, 0, 0, 0));
      }

      con_mesh.synchronize();

      tm.stop(tm_con);
      r3.stop();

      r2.stop();

    }

    if (comm_type::persistent) {
      tm.start(tm_con, "cleanup-persistent-comm");
      comm.cleanup_persistent_comm();
      tm.stop(tm_con);
    }

    comm.barrier();

    tm_total.stop(tm_con);

    r1.stop();

    print_timer(comminfo, tm);
    print_timer(comminfo, tm_total);
  }

  tm.clear();
  tm_total.clear();

  // print_proc_memory_stats(comminfo);
}

} // namespace COMB

#endif // _DO_CYCLES_HPP

