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

#ifndef _DO_CYCLES_HPP
#define _DO_CYCLES_HPP

#include <type_traits>

#include "comb.hpp"
#include "CommFactory.hpp"

namespace COMB {

template < typename pol_comm, typename pol_mesh, typename pol_many, typename pol_few >
bool should_do_cycles(CommContext<pol_comm>& con_comm,
                      ExecContext<pol_mesh>& con_mesh, AllocatorInfo& aloc_mesh,
                      ExecContext<pol_many>& con_many, AllocatorInfo& aloc_many,
                      ExecContext<pol_few>& con_few,  AllocatorInfo& aloc_few)
{
  return aloc_mesh.available() // && aloc_many.available() && aloc_few.available()
      && aloc_many.accessible(con_comm) && aloc_few.accessible(con_comm)
      && aloc_mesh.accessible(con_mesh)
      && aloc_mesh.accessible(con_many) && aloc_many.accessible(con_many)
      && aloc_mesh.accessible(con_few)  && aloc_few.accessible(con_few) ;
}

template < typename pol_comm, typename pol_mesh, typename pol_many, typename pol_few >
void do_cycles(CommContext<pol_comm>& con_comm_in,
               CommInfo& comm_info, MeshInfo& info,
               IdxT num_vars, IdxT ncycles,
               ExecContext<pol_mesh>& con_mesh, COMB::Allocator& aloc_mesh,
               ExecContext<pol_many>& con_many, COMB::Allocator& aloc_many,
               ExecContext<pol_few>& con_few,  COMB::Allocator& aloc_few,
               Timer& tm, Timer& tm_total)
{
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
      CommFactory factory(comminfo);

      for (IdxT i = 0; i < num_vars; ++i) {

        vars.push_back(MeshData(info, aloc_mesh));

        vars[i].allocate();

        DataT* data = vars[i].data();
        IdxT totallen = info.totallen;

        con_mesh.for_all(0, totallen,
                            detail::set_n1(data));

        factory.add_var(vars[i]);

        con_mesh.synchronize();
      }

      factory.populate(comm, con_many, con_few);
    }

    tm_total.stop(tm_con);

    comm.barrier();

    Range r1("test correctness", Range::indigo);

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

        con_mesh.for_all_3d(0, klen,
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
              FGPRINTF(FileGroup::proc, "%p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
            }
            // FGPRINTF(FileGroup::proc, "%p[%i] = %f\n", data, zone, 1.0);
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

      //   con_mesh.for_all_3d(0, klen,
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
      //           FGPRINTF(FileGroup::proc, "%p %i zone %i(%i %i %i) g%i(%i %i %i) = %f expected %f next %f\n", data, branchid, zone, i, j, k, zone_global, iglobal, jglobal, kglobal, found, expected, next);
      //         }
      //         // FGPRINTF(FileGroup::proc, "%p[%i] = %f\n", data, zone, 1.0);
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

        con_mesh.for_all_3d(0, klen,
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
            // FGPRINTF(FileGroup::proc, "%p[%i] = %f\n", data, zone, 1.0);
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

        con_mesh.for_all_3d(kmin, kmax,
                               jmin, jmax,
                               imin, imax,
                               detail::set_1(ilen, ijlen, data));
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

        con_mesh.for_all_3d(0, klen,
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
          //   FGPRINTF(FileGroup::proc, "zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
          // }
          //FGPRINTF(FileGroup::proc, "%p[%i] = %f\n", data, zone, 1.0);
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

        con_mesh.for_all_3d(0, klen,
                               0, jlen,
                               0, ilen,
                               detail::reset_1(ilen, ijlen, data, imin, jmin, kmin, imax, jmax, kmax));
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


#ifdef COMB_ENABLE_MPI

template < typename comm_pol >
void do_cycles_mpi_type(std::true_type const&,
                        CommContext<comm_pol>& con_comm,
                        CommInfo& comminfo, MeshInfo& info,
                        COMB::ExecContexts& exec,
                        AllocatorInfo& mesh_aloc,
                        COMB::ExecutorsAvailable& exec_avail,
                        IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  if (exec_avail.seq && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
  if (exec_avail.omp && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
  if (exec_avail.cuda && exec_avail.mpi_type && exec_avail.mpi_type && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.mpi_type, mesh_aloc, exec.mpi_type, mesh_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), exec.mpi_type, mesh_aloc.allocator(), tm, tm_total);
#endif
}

template < typename comm_pol >
void do_cycles_mpi_type(std::false_type const&,
                        CommContext<comm_pol>&,
                        CommInfo&, MeshInfo&,
                        COMB::ExecContexts&,
                        AllocatorInfo&,
                        COMB::ExecutorsAvailable&,
                        IdxT, IdxT, Timer&, Timer&)
{
}

#endif

template < typename comm_pol >
void do_cycles_allocator(CommContext<comm_pol>& con_comm,
                         CommInfo& comminfo, MeshInfo& info,
                         COMB::ExecContexts& exec,
                         AllocatorInfo& mesh_aloc,
                         AllocatorInfo& cpu_many_aloc, AllocatorInfo& cpu_few_aloc,
                         AllocatorInfo& cuda_many_aloc, AllocatorInfo& cuda_few_aloc,
                         COMB::ExecutorsAvailable& exec_avail,
                         IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.allocator().name());
  Range r0(name, Range::blue);

  if (exec_avail.seq && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.seq, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.seq, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
  if (exec_avail.omp && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

  if (exec_avail.omp && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

  if (exec_avail.omp && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.omp, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.omp, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

#ifdef COMB_ENABLE_CUDA
  if (exec_avail.cuda && exec_avail.seq && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.seq, cpu_many_aloc, exec.seq, cpu_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.seq, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

#ifdef COMB_ENABLE_OPENMP
  if (exec_avail.cuda && exec_avail.omp && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.seq, cpu_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

  if (exec_avail.cuda && exec_avail.omp && exec_avail.omp && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.omp, cpu_many_aloc, exec.omp, cpu_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.omp, cpu_many_aloc.allocator(), exec.omp, cpu_few_aloc.allocator(), tm, tm_total);
#endif

  if (exec_avail.cuda && exec_avail.cuda && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.seq, cpu_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

  if (exec_avail.cuda && exec_avail.cuda && exec_avail.cuda && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda, cuda_many_aloc, exec.cuda, cuda_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda, cuda_many_aloc.allocator(), exec.cuda, cuda_few_aloc.allocator(), tm, tm_total);

  {
    if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_batch && exec_avail.cuda_batch && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_persistent && exec_avail.cuda_persistent && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);


    SetReset<bool> sr_gs(get_batch_always_grid_sync(), false);

    if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_batch_fewgs && exec_avail.cuda_batch_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_batch, cuda_many_aloc, exec.cuda_batch, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_batch, cuda_many_aloc.allocator(), exec.cuda_batch, cuda_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.seq, cpu_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

    if (exec_avail.cuda && exec_avail.cuda_persistent_fewgs && exec_avail.cuda_persistent_fewgs && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_persistent, cuda_many_aloc, exec.cuda_persistent, cuda_few_aloc))
      do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_persistent, cuda_many_aloc.allocator(), exec.cuda_persistent, cuda_few_aloc.allocator(), tm, tm_total);
  }

#ifdef COMB_ENABLE_CUDA_GRAPH
  if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.seq && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.seq, cpu_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.seq, cpu_few_aloc.allocator(), tm, tm_total);

  if (exec_avail.cuda && exec_avail.cuda_graph && exec_avail.cuda_graph && should_do_cycles(con_comm, exec.cuda, mesh_aloc, exec.cuda_graph, cuda_many_aloc, exec.cuda_graph, cuda_few_aloc))
    do_cycles(con_comm, comminfo, info, num_vars, ncycles, exec.cuda, mesh_aloc.allocator(), exec.cuda_graph, cuda_many_aloc.allocator(), exec.cuda_graph, cuda_few_aloc.allocator(), tm, tm_total);
#endif
#else
  COMB::ignore_unused(cuda_many_aloc, cuda_few_aloc);
#endif

#ifdef COMB_ENABLE_MPI
  do_cycles_mpi_type(typename std::conditional<comm_pol::use_mpi_type, std::true_type, std::false_type>::type{},
      con_comm, comminfo, info, exec, mesh_aloc, exec_avail, num_vars, ncycles, tm, tm_total);
#endif
}


template < typename comm_pol >
void do_cycles_allocators(CommContext<comm_pol>& con_comm,
                          CommInfo& comminfo, MeshInfo& info,
                          COMB::ExecContexts& exec,
                          Allocators& alloc,
                          AllocatorInfo& cpu_many_aloc, AllocatorInfo& cpu_few_aloc,
                          AllocatorInfo& cuda_many_aloc, AllocatorInfo& cuda_few_aloc,
                          COMB::ExecutorsAvailable& exec_avail,
                          IdxT num_vars, IdxT ncycles, Timer& tm, Timer& tm_total)
{
  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.host,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      exec_avail,
                      num_vars, ncycles, tm, tm_total);

#ifdef COMB_ENABLE_CUDA

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_hostpinned,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      exec_avail,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_device,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      exec_avail,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      exec_avail,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed_host_preferred,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      exec_avail,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed_host_preferred_device_accessed,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      exec_avail,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed_device_preferred,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      exec_avail,
                      num_vars, ncycles, tm, tm_total);

  do_cycles_allocator(con_comm,
                      comminfo, info,
                      exec,
                      alloc.cuda_managed_device_preferred_host_accessed,
                      cpu_many_aloc, cpu_few_aloc,
                      cuda_many_aloc, cuda_few_aloc,
                      exec_avail,
                      num_vars, ncycles, tm, tm_total);

#endif // COMB_ENABLE_CUDA

}

} // namespace COMB

#endif // _DO_CYCLES_HPP

