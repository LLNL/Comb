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

#include "comb.hpp"

#include "comm_pol_mock.hpp"
#include "comm_pol_mpi.hpp"
#include "CommFactory.hpp"

namespace COMB {

template < typename pol_comm, typename pol_mesh, typename pol_many, typename pol_few >
void do_cycles_basic(CommContext<pol_comm>& con_comm_in,
                     CommInfo& comm_info, MeshInfo& info,
                     IdxT num_vars, IdxT ncycles,
                     ExecContext<pol_mesh>& con_mesh, COMB::Allocator& aloc_mesh,
                     ExecContext<pol_many>& con_many, COMB::Allocator& aloc_many,
                     ExecContext<pol_few>& con_few,  COMB::Allocator& aloc_few,
                     Timer& tm, Timer& tm_total)
{
  static_assert(std::is_same<pol_many, pol_few>::value, "do_cycles_basic expects pol_many and pol_few to be the same");
  static_assert(std::is_same<pol_many, seq_pol>::value
#ifdef COMB_ENABLE_CUDA
             || std::is_same<pol_many, cuda_pol>::value
#endif
#ifdef COMB_ENABLE_HIP
             || std::is_same<pol_many, hip_pol>::value
#endif
               ,"do_cycles_basic expects pol_many to be seq_pol, cuda_pol, or hip_pol");

  CPUContext tm_con;
  tm_total.clear();
  tm.clear();

  char test_name[1024] = ""; snprintf(test_name, 1024, "Basic\nComm %s Mesh %s %s Buffers %s %s %s %s",
                                                        pol_comm::get_name(),
                                                        pol_mesh::get_name(), aloc_mesh.name(),
                                                        pol_many::get_name(), aloc_many.name(), pol_few::get_name(), aloc_few.name());
  fgprintf(FileGroup::all, "Starting test %s\n", test_name);

  {
    Range r0(test_name, Range::orange);

    // make a copy of comminfo to duplicate the MPI communicator
    CommInfo comminfo(comm_info);

    using comm_type = Comm<pol_many, pol_few, pol_comm>;
    using policy_comm  = typename comm_type::policy_comm;
    using recv_message_type = typename comm_type::recv_message_type;
    using send_message_type = typename comm_type::send_message_type;

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
      CommFactory factory(comminfo);

      for (IdxT i = 0; i < num_vars; ++i) {

        vars.emplace_back(info, aloc_mesh);

        MeshData& var = vars.back();

        // allocate variable
        var.allocate();

        // initialize variable
        DataT* data = var.data();
        IdxT totallen = info.totallen;

        con_mesh.for_all(totallen,
                         detail::set_n1(data));
        con_mesh.synchronize();

        // add variable to comm
        factory.add_var(var);
      }

      factory.populate(comm, con_many, con_few);
    }

    // do_cycles_basic expects that all sends and receives have_many (use pol_many)
    assert(comm.m_recvs.message_group_few.messages.size() == 0);
    assert(comm.m_sends.message_group_few.messages.size() == 0);

    tm_total.stop(tm_con);

   /**************************************************************************
    **************************************************************************
    *
    * Perform test communication steps to ensure communication gives
    * the right answer
    *
    **************************************************************************
    **************************************************************************/

    comm.barrier();

    Range r1("test comm", Range::indigo);

    tm_total.start(tm_con, "test-comm");

    IdxT ntestcycles = std::max(IdxT{1}, ncycles/IdxT{10});
    for (IdxT test_cycle = 0; test_cycle < ntestcycles; ++test_cycle) { // test comm

      Range r2("cycle", Range::cyan);

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
              FGPRINTF(FileGroup::proc, "%p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
            }
            // LOGPRINTF("%p[%i] = %f\n", data, zone, 1.0);
            assert(found == expected);
          }
          data[zone] = next;
        });
      }

      con_mesh.synchronize();

      // tm.stop(tm_con);
      r3.restart("post-recv", Range::pink);
      // tm.start(tm_con, "post-recv");

     /************************************************************************
      *
      * Allocate receive buffers and post receives
      *
      ************************************************************************/
      comm.postRecv(con_many, con_few);

      // tm.stop(tm_con);
      r3.restart("post-send", Range::pink);
      // tm.start(tm_con, "post-send");

     /************************************************************************
      *
      * Allocate send buffers, pack, and post sends
      *
      ************************************************************************/
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
      //           FGPRINTF(FileGroup::proc, "%p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
      //         }
      //         // LOGPRINTF("%p[%i] = %f\n", data, zone, 1.0);
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

     /************************************************************************
      *
      * Wait on receives, unpack, and deallocate receive buffers
      *
      ************************************************************************/
      comm.waitRecv(con_many, con_few);

      // tm.stop(tm_con);
      r3.restart("wait-send", Range::pink);
      // tm.start(tm_con, "wait-send");

     /************************************************************************
      *
      * Wait on sends and deallocate send buffers
      *
      ************************************************************************/
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
              FGPRINTF(FileGroup::proc, "%p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
            }
            // LOGPRINTF("%p[%i] = %f\n", data, zone, 1.0);
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

    comm.barrier();

    tm_total.stop(tm_con);

    tm.clear();


   /**************************************************************************
    **************************************************************************
    *
    * Perform "real" communication steps to benchmark performance
    *
    **************************************************************************
    **************************************************************************/

    r1.restart("bench comm", Range::magenta);

    tm_total.start(tm_con, "bench-comm");

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

     /************************************************************************
      *
      * Allocate receive buffers and post receives
      *
      ************************************************************************/
      // comm.postRecv(con_many, con_few);
      {
        // LOGPRINTF("posting receives\n");

        IdxT num_recvs = comm.m_recvs.message_group_many.messages.size();

        comm.m_recvs.requests.resize(num_recvs, comm.con_comm.recv_request_null());

        for (IdxT i = 0; i < num_recvs; ++i) {
          recv_message_type* message = &comm.m_recvs.message_group_many.messages[i];

          comm.m_recvs.message_group_many.allocate(con_many, comm.con_comm, &message, 1, ::detail::Async::no);
          comm.m_recvs.message_group_many.Irecv(con_many, comm.con_comm, &message, 1, ::detail::Async::no, &comm.m_recvs.requests[i]);
        }
      }

      tm.stop(tm_con);
      r3.restart("post-send", Range::pink);
      tm.start(tm_con, "post-send");

     /************************************************************************
      *
      * Allocate send buffers, pack, and post sends
      *
      ************************************************************************/
      // comm.postSend(con_many, con_few);
      {
        // LOGPRINTF("posting sends\n");

        const IdxT num_sends = comm.m_sends.message_group_many.messages.size();

        comm.m_sends.requests.resize(num_sends, comm.con_comm.send_request_null());

        for (IdxT i = 0; i < num_sends; ++i) {
          send_message_type* message = &comm.m_sends.message_group_many.messages[i];

          comm.m_sends.message_group_many.allocate(con_many, comm.con_comm, &message, 1, ::detail::Async::no);
          comm.m_sends.message_group_many.pack(con_many, comm.con_comm, &message, 1, ::detail::Async::no);
          comm.m_sends.message_group_many.wait_pack_complete(con_many, comm.con_comm, &message, 1, ::detail::Async::no);
          comm.m_sends.message_group_many.Isend(con_many, comm.con_comm, &message, 1, ::detail::Async::no, &comm.m_sends.requests[i]);
        }
      }

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
          //LOGPRINTF("%p[%i] = %f\n", data, zone, 1.0);
          data[zone] = next;
        });
      }
      */

      r3.start("wait-recv", Range::pink);
      tm.start(tm_con, "wait-recv");

     /************************************************************************
      *
      * Wait on receives, unpack, and deallocate receive buffers
      *
      ************************************************************************/
      // comm.waitRecv(con_many, con_few);
      {
        // LOGPRINTF("waiting receives\n");

        IdxT num_recvs = comm.m_recvs.message_group_many.messages.size();

        typename policy_comm::recv_status_type status = comm.con_comm.recv_status_null();

        IdxT num_done = 0;
        while (num_done < num_recvs) {

          IdxT idx = recv_message_type::wait_recv_any(comm.con_comm, num_recvs, &comm.m_recvs.requests[0], &status);

          recv_message_type* message = &comm.m_recvs.message_group_many.messages[idx];

          comm.m_recvs.message_group_many.unpack(con_many, comm.con_comm, &message, 1, ::detail::Async::no);
          comm.m_recvs.message_group_many.deallocate(con_many, comm.con_comm, &message, 1, ::detail::Async::no);

          num_done += 1;

        }

        comm.m_recvs.requests.clear();

        con_many.synchronize();
      }

      tm.stop(tm_con);
      r3.restart("wait-send", Range::pink);
      tm.start(tm_con, "wait-send");

     /************************************************************************
      *
      * Wait on sends and deallocate send buffers
      *
      ************************************************************************/
      // comm.waitSend(con_many, con_few);
      {
        // LOGPRINTF("posting sends\n");

        IdxT num_sends = comm.m_sends.message_group_many.messages.size();

        typename policy_comm::send_status_type status = comm.con_comm.send_status_null();

        IdxT num_done = 0;
        while (num_done < num_sends) {

          IdxT idx = send_message_type::wait_send_any(comm.con_comm, num_sends, &comm.m_sends.requests[0], &status);

          send_message_type* message = &comm.m_sends.message_group_many.messages[idx];

          comm.m_sends.message_group_many.deallocate(con_many, comm.con_comm, &message, 1, ::detail::Async::no);

          num_done += 1;
        }

        comm.m_sends.requests.clear();
      }

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


void test_cycles_basic(CommInfo& comminfo, MeshInfo& info,
                       COMB::Executors& exec,
                       COMB::Allocators& alloc,
                       IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
#ifdef COMB_ENABLE_MPI
  CommContext<mpi_pol> con_comm{exec.base_mpi.get()};
#else
  CommContext<mock_pol> con_comm{exec.base_cpu.get()};
#endif

  {
    // mpi/mock sequential exec host memory test

    do_cycles_basic(con_comm,
                    comminfo, info,
                    num_vars, ncycles,
                    exec.seq.get(), alloc.host.allocator(),
                    exec.seq.get(), alloc.host.allocator(),
                    exec.seq.get(), alloc.host.allocator(),
                    tm, tm_total);
  }

#ifdef COMB_ENABLE_CUDA
  {
    // mpi cuda exec cuda memory test

    do_cycles_basic(con_comm,
                    comminfo, info,
                    num_vars, ncycles,
                    exec.cuda.get(), alloc.cuda_managed.allocator(),
                    exec.cuda.get(), alloc.cuda_hostpinned.allocator(),
                    exec.cuda.get(), alloc.cuda_hostpinned.allocator(),
                    tm, tm_total);
  }
#endif

#ifdef COMB_ENABLE_HIP
  {
    // mpi hip exec hip memory test

    do_cycles_basic(con_comm,
                    comminfo, info,
                    num_vars, ncycles,
                    exec.hip.get(), alloc.hip_managed.allocator(),
                    exec.hip.get(), alloc.hip_hostpinned.allocator(),
                    exec.hip.get(), alloc.hip_hostpinned.allocator(),
                    tm, tm_total);
  }
#endif

}

} // namespace COMB
