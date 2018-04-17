#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cctype>

#include <mpi.h>

#include "memory.cuh"
#include "for_all.cuh"
#include "profiling.cuh"
#include "MeshInfo.cuh"
#include "MeshData.cuh"
#include "comm.cuh"
#include "CommFactory.cuh"

namespace detail {

  struct set_n1 {
     DataT* data;
     set_n1(DataT* data_) : data(data_) {}
     HOST DEVICE
     void operator()(IdxT i, IdxT) const {
       IdxT zone = i;
       //FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
       data[zone] = -1.0;
     }
  };

  struct set_1 {
     IdxT ilen, ijlen;
     DataT* data;
     set_1(IdxT ilen_, IdxT ijlen_, DataT* data_) : ilen(ilen_), ijlen(ijlen_), data(data_) {}
     HOST DEVICE
     void operator()(IdxT k, IdxT j, IdxT i, IdxT idx) const {
       IdxT zone = i + j * ilen + k * ijlen;
       //FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
       data[zone] = 1.0;
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
     HOST DEVICE
     void operator()(IdxT k, IdxT j, IdxT i, IdxT idx) const {
       IdxT zone = i + j * ilen + k * ijlen;
       DataT expected, found, next;
       if (k >= kmin && k < kmax &&
           j >= jmin && j < jmax &&
           i >= imin && i < imax) {
         expected = 1.0; found = data[zone]; next = 1.0;
       } else {
         expected = 0.0; found = data[zone]; next = -1.0;
       }
       //if (found != expected) FPRINTF(stdout, "zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
       //FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
       data[zone] = next;
     }
  };

} // namespace detail

template < typename pol_loop, typename pol_many, typename pol_few >
void do_cycles(CommInfo& comm_info, MeshInfo& info, IdxT num_vars, IdxT ncycles, Allocator& aloc_mesh, Allocator& aloc_many, Allocator& aloc_few, Timer& tm)
{
    tm.clear();

    char rname[1024] = ""; snprintf(rname, 1024, "Buffers %s %s %s %s", pol_many::name, aloc_many.name(), pol_few::name, aloc_few.name());
    char test_name[1024] = ""; snprintf(test_name, 1024, "Mesh %s %s %s", pol_loop::name, aloc_mesh.name(), rname);
    FPRINTF(stdout, "Starting test %s\n", test_name);

    Range r0(rname, Range::orange);
    
    comm_info.barrier();

    tm.start("start-up");

    std::vector<MeshData> vars;
    vars.reserve(num_vars);
    
    Comm<pol_many, pol_few> comm(comm_info, aloc_many, aloc_few);
    
    {
      CommFactory factory(comm_info);
      
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

    tm.stop();
    
    { // test comm

      Range r1("test comm", Range::magenta);

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
      IdxT ijlen = info.stride[2];
      IdxT ijlen_global = ilen_global * jlen_global;
      
      
      Range r2("pre-comm", Range::red);
      // tm.start("pre-comm");

      for (IdxT i = 0; i < num_vars; ++i) {
      
        DataT* data = vars[i].data();
        IdxT var_i = i;
      
        for_all_3d(pol_loop{}, 0, klen,
                               0, jlen,
                               0, ilen,
                               [=] HOST DEVICE (IdxT k, IdxT j, IdxT i, IdxT idx) {
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
          if (k >= kmin && k < kmax &&
              j >= jmin && j < jmax &&
              i >= imin && i < imax) {
            next = zone_global + var_i;
          } else if (iglobal < 0 || iglobal >= ilen_global ||
                     jglobal < 0 || jglobal >= jlen_global ||
                     kglobal < 0 || kglobal >= klen_global) {
            next = -zone_global - var_i;
          } else {
            next = -zone_global - var_i;
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
      
      
      for (IdxT i = 0; i < num_vars; ++i) {
      
        DataT* data = vars[i].data();
        IdxT var_i = i;
        
        for_all_3d(pol_loop{}, 0, klen,
                               0, jlen,
                               0, ilen,
                               [=] HOST DEVICE (IdxT k, IdxT j, IdxT i, IdxT idx) {
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
          if (k >= kmin && k < kmax &&
              j >= jmin && j < jmax &&
              i >= imin && i < imax) {
            expected = zone_global + var_i;  found = data[zone]; next = -1.0;
          } else if (iglobal < 0 || iglobal >= ilen_global ||
                     jglobal < 0 || jglobal >= jlen_global ||
                     kglobal < 0 || kglobal >= klen_global) {
            expected = -zone_global - var_i; found = data[zone]; next = -zone_global - var_i;
          } else {
            expected = -zone_global - var_i; found = data[zone]; next = 1.0;
          }
          //if (found != expected) FPRINTF(stdout, "zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
          //FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
          assert(found == expected);
          data[zone] = next;
        });
      }
      

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
        IdxT var_i = i;
        
        for_all_3d(pol_loop{}, 0, klen,
                               0, jlen,
                               0, ilen,
                               [=] HOST DEVICE (IdxT k, IdxT j, IdxT i, IdxT idx) {
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
          if (k >= kmin && k < kmax &&
              j >= jmin && j < jmax &&
              i >= imin && i < imax) {
            expected = -1.0;                 found = data[zone]; next = 1.0;
          } else if (iglobal < 0 || iglobal >= ilen_global ||
                     jglobal < 0 || jglobal >= jlen_global ||
                     kglobal < 0 || kglobal >= klen_global) {
            expected = -zone_global - var_i; found = data[zone]; next = -1.0;
          } else {
            expected = zone_global + var_i;  found = data[zone]; next = -1.0;
          }
          //if (found != expected) FPRINTF(stdout, "zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
          //FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
          assert(found == expected);
          data[zone] = next;
        });
      }
      
      synchronize(pol_loop{});

      // tm.stop();
      r2.stop();

    }

    for(IdxT cycle = 0; cycle < ncycles; cycle++) {

      Range r1("cycle", Range::yellow);

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
      
      
      Range r2("pre-comm", Range::red);
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
      r2.restart("post-recv", Range::pink);
      tm.start("post-recv");
      
      comm.postRecv();

      tm.stop();
      r2.restart("post-send", Range::pink);
      tm.start("post-send");

      comm.postSend();
      
      tm.stop();
      r2.stop();
      
      /*
      for (IdxT i = 0; i < num_vars; ++i) {
      
        DataT* data = vars[i].data();
        
        for_all_3d(pol_loop{}, 0, klen,
                               0, jlen,
                               0, ilen,
                               [=] HOST DEVICE (IdxT k, IdxT j, IdxT i, IdxT idx) {
          IdxT zone = i + j * ilen + k * ijlen;
          DataT expected, found, next;
          if (k >= kmin && k < kmax &&
              j >= jmin && j < jmax &&
              i >= imin && i < imax) {
            expected = 1.0; found = data[zone]; next = 1.0;
          } else {
            expected = -1.0; found = data[zone]; next = -1.0;
          }
          if (found != expected) FPRINTF(stdout, "zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
          //FPRINTF(stdout, "%p[%i] = %f\n", data, zone, 1.0);
          data[zone] = next;
        });
      }
      */

      r2.start("wait-recv", Range::pink);
      tm.start("wait-recv");

      comm.waitRecv();

      tm.stop();
      r2.restart("wait-send", Range::pink);
      tm.start("wait-send");

      comm.waitSend();

      tm.stop();
      r2.restart("post-comm", Range::red);
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
      r2.stop();

    }

    tm.print();
    tm.clear();
}
 
template < typename pol_type >
void prime_allocator(pol_type const& pol, Allocator& aloc, Timer& tm, IdxT num_vars, IdxT len)
{
  DataT** vars = new DataT*[num_vars];
  
  tm.start(aloc.name());
    
  for (IdxT i = 0; i < num_vars; ++i) {
    vars[i] = (DataT*)aloc.allocate(len*sizeof(DataT));
  }
  
  for (IdxT i = 0; i < num_vars; ++i) {
  
    DataT* data = vars[i];
    
    for_all(pol, 0, len, [=] HOST DEVICE (IdxT, IdxT idx) {
      data[idx] = 0.0;
    });
  }
  
  for (IdxT i = 0; i < num_vars; ++i) {
    aloc.deallocate(vars[i]);
  }
  
  tm.stop();
  
  delete[] vars;
}

int main(int argc, char** argv)
{
  int required = MPI_THREAD_SINGLE;
  int provided = detail::MPI::Init_thread(&argc, &argv, required);
  
  CommInfo comminfo;
  
  if (required != provided) {
    comminfo.abort_master("Didn't receive MPI thread support required %i provided %i.\n", required, provided);
  }
  
  comminfo.print_master("Started rank %i of %i\n", comminfo.rank, comminfo.size);

  cudaCheck(cudaDeviceSynchronize());  

  IdxT sizes[3] = {0, 0, 0};
  int divisions[3] = {0, 0, 0};
  int periodic[3] = {0, 0, 0};
  IdxT ghost_width = 1;
  IdxT num_vars = 1;
  IdxT ncycles = 5;
  
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
              comminfo.cutoff = static_cast<IdxT>(atoll(argv[++i]));
            } else {
              comminfo.warn_master("No argument to sub-option, ignoring %s.\n", argv[i]);
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
                comminfo.warn_master("Invalid argument to sub-option, ignoring %s.\n", argv[i-1]);
              }
            } else {
              comminfo.warn_master("No argument to sub-option, ignoring %s.\n", argv[i]);
            }
          } else {
            comminfo.warn_master("Invalid argument to option, ignoring %s.\n", argv[i-1]);
          }
        } else {
          comminfo.warn_master("No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "ghost") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          ghost_width = static_cast<IdxT>(atoll(argv[++i]));
        } else {
          comminfo.warn_master("No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "vars") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          num_vars = static_cast<IdxT>(atoll(argv[++i]));
        } else {
          comminfo.warn_master("No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "cycles") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          ncycles = static_cast<IdxT>(atoll(argv[++i]));
        } else {
          comminfo.warn_master("No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "periodic") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          int ret = sscanf(argv[++i], "%d_%d_%d", &periodic[0], &periodic[1], &periodic[2]);
          if (ret == 1) {
            periodic[1] = periodic[0];
            periodic[2] = periodic[0];
          } else if (ret != 3) {
            periodic[0] = 0;
            periodic[1] = 0;
            periodic[2] = 0;
            comminfo.warn_master("Invalid arguments to option, ignoring %s.\n", argv[i-1]);
          }
          periodic[0] = periodic[0] ? 1 : 0;
          periodic[1] = periodic[1] ? 1 : 0;
          periodic[2] = periodic[2] ? 1 : 0;
        } else {
          comminfo.warn_master("No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "divide") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          int ret = sscanf(argv[++i], "%d_%d_%d", &divisions[0], &divisions[1], &divisions[2]);
          if (ret != 3 || divisions[0] < 1 || divisions[1] < 1 || divisions[2] < 1) {
            divisions[0] = 0;
            divisions[1] = 0;
            divisions[2] = 0;
            comminfo.warn_master("Invalid arguments to option, ignoring %s.\n", argv[i-1]);
          }
        } else {
          comminfo.warn_master("No argument to option, ignoring %s.\n", argv[i]);
        }
      } else {
        comminfo.warn_master("Unknown option, ignoring %s.\n", argv[i]); 
      }
    } else if (std::isdigit(argv[i][0]) && s < 1) {
      long read_sizes[3] {0, 0, 0};
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
        // set sizes invalid
        sizes[0] = 0;
        sizes[1] = 0;
        sizes[2] = 0;
        comminfo.warn_master("Invalid sizes argument, ignoring %s.\n", argv[i]);
      }
    } else {
      comminfo.warn_master("Invalid argument, ignoring %s.\n", argv[i]);
    }
  }
  
  if (ncycles <= 0) {
    comminfo.abort_master("Invalid cycles argument.\n");
  } else if (num_vars <= 0) {
    comminfo.abort_master("Invalid vars argument.\n");
  } else if (ghost_width <= 0) {
    comminfo.abort_master("Invalid ghost argument.\n");
  } else if ( (divisions[0] != 0 || divisions[1] != 0 || divisions[2] != 0)
           && comminfo.size != divisions[0] * divisions[1] * divisions[2]) {
    comminfo.abort_master("Invalid mesh divisions\n");
  }
  
  GlobalMeshInfo global_info(sizes, comminfo.size, divisions, periodic, ghost_width);
  
  // create cartesian communicator and get rank
  comminfo.cart.create(global_info.divisions, global_info.periodic);
  
  MeshInfo info = MeshInfo::get_local(global_info, comminfo.cart.coords);
  
  // print info about problem setup
  comminfo.print_master("Do %s communication\n", comminfo.mock_communication ? "mock" : "real");
  comminfo.print_master("Message policy cutoff %i\n", comminfo.cutoff);
  comminfo.print_master("Post Recv using %s method\n", CommInfo::method_str(comminfo.post_recv_method));
  comminfo.print_master("Post Send using %s method\n", CommInfo::method_str(comminfo.post_send_method));
  comminfo.print_master("Wait Recv using %s method\n", CommInfo::method_str(comminfo.wait_recv_method));
  comminfo.print_master("Wait Send using %s method\n", CommInfo::method_str(comminfo.wait_send_method));
  comminfo.print_master("Num cycles  %i\n", ncycles);
  comminfo.print_master("Num cycles  %i\n", ncycles);
  comminfo.print_master("Num vars    %i\n", num_vars);
  comminfo.print_master("ghost_width %i\n", info.ghost_width);
  comminfo.print_master("size      %8i %8i %8i\n", global_info.sizes[0],       global_info.sizes[1],       global_info.sizes[2]);
  comminfo.print_master("divisions %8i %8i %8i\n", comminfo.cart.divisions[0], comminfo.cart.divisions[1], comminfo.cart.divisions[2]);
  comminfo.print_master("periodic  %8i %8i %8i\n", comminfo.cart.periodic[0],  comminfo.cart.periodic[1],  comminfo.cart.periodic[2]);
  comminfo.print_master("division map\n", comminfo.cart.periodic[0], comminfo.cart.periodic[1], comminfo.cart.periodic[2]);
  // print division map
  IdxT max_cuts = std::max(std::max(comminfo.cart.divisions[0], comminfo.cart.divisions[1]), comminfo.cart.divisions[2]);
  for (IdxT ci = 0; ci <= max_cuts; ++ci) {
    int division_coords[3] {-1, -1, -1};
    if (ci <= comminfo.cart.divisions[0]) {
      division_coords[0] = ci * (sizes[0] / comminfo.cart.divisions[0]) + std::min(ci, sizes[0] % comminfo.cart.divisions[0]);
    }
    if (ci <= comminfo.cart.divisions[0]) {
      division_coords[1] = ci * (sizes[1] / comminfo.cart.divisions[1]) + std::min(ci, sizes[1] % comminfo.cart.divisions[1]);
    }
    if (ci <= comminfo.cart.divisions[0]) {
      division_coords[2] = ci * (sizes[2] / comminfo.cart.divisions[2]) + std::min(ci, sizes[2] % comminfo.cart.divisions[2]);
    }
    comminfo.print_master("map       %8i %8i %8i\n", division_coords[0], division_coords[1], division_coords[2] );
  }
  
  HostAllocator host_alloc;
  HostPinnedAllocator hostpinned_alloc;
  DeviceAllocator device_alloc;
  ManagedAllocator managed_alloc;
  ManagedHostPreferredAllocator managed_host_preferred_alloc;
  ManagedHostPreferredDeviceAccessedAllocator managed_host_preferred_device_accessed_alloc;
  ManagedDevicePreferredAllocator managed_device_preferred_alloc;
  ManagedDevicePreferredHostAccessedAllocator managed_device_preferred_host_accessed_alloc;
  
  Timer tm(1024);

  // warm-up memory pools
  {
    Range r("Memmory pool init", Range::green);
    
    FPRINTF(stdout, "Starting up memory pools\n");
    
    prime_allocator(seq_pol{},        host_alloc,                                   tm, num_vars+1, info.totallen);
    
    prime_allocator(omp_pol{},        hostpinned_alloc,                             tm, num_vars+1, info.totallen);
    
    prime_allocator(cuda_pol{},       device_alloc,                                 tm, num_vars+1, info.totallen);
    
    prime_allocator(cuda_batch_pol{}, managed_alloc,                                tm, num_vars+1, info.totallen);
    
    prime_allocator(seq_pol{},        managed_host_preferred_alloc,                 tm, num_vars+1, info.totallen);
    
    prime_allocator(omp_pol{},        managed_host_preferred_device_accessed_alloc, tm, num_vars+1, info.totallen);
    
    prime_allocator(cuda_pol{},       managed_device_preferred_alloc,               tm, num_vars+1, info.totallen);
    
    prime_allocator(cuda_batch_pol{}, managed_device_preferred_host_accessed_alloc, tm, num_vars+1, info.totallen);

    tm.print();
    tm.clear();

  }

  // host allocated
  {
    Allocator& mesh_aloc = host_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);
  }

  // host pinned allocated
  {
    Allocator& mesh_aloc = hostpinned_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // device allocated
  {
    Allocator& mesh_aloc = device_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    // do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);
    
    do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);
 
    if (comminfo.mock_communication) {
      do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, device_alloc, device_alloc, tm);
 
      do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, device_alloc, device_alloc, tm);
    }
  }

  // managed allocated
  {
    Allocator& mesh_aloc = managed_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed host preferred allocated
  {
    Allocator& mesh_aloc = managed_host_preferred_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed host preferred device accessed allocated
  {
    Allocator& mesh_aloc = managed_host_preferred_device_accessed_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed device preferred allocated
  {
    Allocator& mesh_aloc = managed_device_preferred_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed device preferred host accessed allocated
  {
    Allocator& mesh_aloc = managed_device_preferred_host_accessed_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);

    do_cycles<cuda_pol, cuda_batch_pol, seq_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, tm);
    
    do_cycles<cuda_pol, cuda_batch_pol, cuda_batch_pol>(comminfo, info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  comminfo.cart.disconnect();
  detail::MPI::Finalize();
  return 0;
}

