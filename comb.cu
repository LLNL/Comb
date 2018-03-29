#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <set>

#include <mpi.h>

#include "memory.cuh"
#include "for_all.cuh"
#include "profiling.cuh"
#include "mesh.cuh"
#include "comm.cuh"

namespace detail {

  struct set_n1 {
     DataT* data;
     set_n1(DataT* data_) : data(data_) {}
     HOST DEVICE
     void operator()(IdxT i, IdxT) const {
       IdxT zone = i;
       //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
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
       //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
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
       //if (found != expected) printf("zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
       //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
       data[zone] = next;
     }
  };

} // namespace detail

template < typename pol_loop, typename pol_face, typename pol_edge, typename pol_corner >
void do_cycles(MeshInfo& info, IdxT num_vars, IdxT ncycles, Allocator& aloc_mesh, Allocator& aloc_face, Allocator& aloc_edge, Allocator& aloc_corner, Timer& tm)
{
    tm.clear();

    char rname[1024] = ""; snprintf(rname, 1024, "Buffers %s %s %s %s %s %s", pol_face::name, aloc_face.name(), pol_edge::name, aloc_edge.name(), pol_corner::name, aloc_corner.name());
    char test_name[1024] = ""; snprintf(test_name, 1024, "Mesh %s %s %s", pol_loop::name, aloc_mesh.name(), rname);
    printf("Starting test %s\n", test_name); fflush(stdout);

    Range r0(rname, Range::orange);

    tm.start("start-up");

    std::vector<MeshData> vars;
    vars.reserve(num_vars);
    
    Comm<pol_face, pol_edge, pol_corner> comm(aloc_face, aloc_edge, aloc_corner);

    for (IdxT i = 0; i < num_vars; ++i) {
    
      vars.push_back(MeshData(info, aloc_mesh));
      
      vars[i].allocate();
      
      DataT* data = vars[i].data();
      IdxT ijklen = info.ijklen;

      for_all(pol_loop{}, 0, ijklen,
                          detail::set_n1(data));

      comm.add_var(vars[i]);
      
      synchronize(pol_loop{});
    }

    tm.stop();

    for(IdxT cycle = 0; cycle < ncycles; cycle++) {

      Range r1("cycle", Range::yellow);

      IdxT imin = info.imin;
      IdxT jmin = info.jmin;
      IdxT kmin = info.kmin;
      IdxT imax = info.imax;
      IdxT jmax = info.jmax;
      IdxT kmax = info.kmax;
      IdxT ilen = info.ilen;
      IdxT jlen = info.jlen;
      IdxT klen = info.klen;
      IdxT ijlen = info.ijlen;
      
      
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

      synchronize(pol_corner{}, pol_edge{}, pol_face{});
      
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
          if (found != expected) printf("zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
          //printf("%p[%i] = %f\n", data, zone, 1.0);
          data[zone] = next;
        });
      }
      */

      r2.start("wait-recv", Range::pink);
      tm.start("wait-recv");

      comm.waitRecv();

      synchronize(pol_corner{}, pol_edge{}, pol_face{});

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
 

int main(int argc, char** argv)
{
  int required = MPI_THREAD_SINGLE;
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, required, &provided);

  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  if (required != provided) {
    fprintf(stderr, "Didn't receive MPI thread support required %i provided %i.\n", required, provided); fflush(stderr);
    MPI_Abort(mpi_comm, 1);
  }

  int comm_rank = -1;
  MPI_Comm_rank(mpi_comm, &comm_rank);
  int comm_size = 0;
  MPI_Comm_size(mpi_comm, &comm_size);

  if (comm_rank == 0) {
    printf("Started rank %i of %i\n", comm_rank, comm_size); fflush(stdout);
  }

  cudaCheck(cudaDeviceSynchronize());  

  IdxT sizes[3] = {0, 0, 0};
  IdxT ghost_width = 1;
  IdxT num_vars = 1;
  IdxT ncycles = 5;
  int periodic[3] {0, 0, 0};
  
  IdxT i = 1;
  IdxT s = 0;
  for(; i < argc; ++i) {
    if (argv[i][0] == '-') {
      // option
      if (strcmp(&argv[i][1], "ghost") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          ghost_width = static_cast<IdxT>(atoll(argv[++i]));
        } else {
          if (comm_rank == 0) {
            fprintf(stderr, "No argument to option, ignoring %s.\n", argv[i]); fflush(stderr);
          }
        }
      } else if (strcmp(&argv[i][1], "vars") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          num_vars = static_cast<IdxT>(atoll(argv[++i]));
        } else {
          if (comm_rank == 0) {
            fprintf(stderr, "No argument to option, ignoring %s.\n", argv[i]); fflush(stderr);
          }
        }
      } else if (strcmp(&argv[i][1], "cycles") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          ncycles = static_cast<IdxT>(atoll(argv[++i]));
        } else {
          if (comm_rank == 0) {
            fprintf(stderr, "No argument to option, ignoring %s.\n", argv[i]); fflush(stderr);
          }
        }
      } else {
        if (comm_rank == 0) {
          fprintf(stderr, "Unknown option, ignoring %s.\n", argv[i]); fflush(stderr);
        }
      }
    } else if ( s < 3 ) {
      // assume got integer
      sizes[s++] = static_cast<IdxT>(atoll(argv[i]));
    } else {
      if (comm_rank == 0) {
        fprintf(stderr, "Too many arguments, ignoring %s.\n", argv[i]); fflush(stderr);
      }
    }
  }
  
  if (s == 1) {
    sizes[1] = sizes[0];
    sizes[2] = sizes[0];
  }
  
  if (ncycles <= 0) {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid cycles argument.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else if (num_vars <= 0) {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid vars argument.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else if (ghost_width <= 0) {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid ghost argument.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  } else if (sizes[0] < ghost_width || sizes[1] < ghost_width || sizes[2] < ghost_width) {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid size arguments.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  
  int cuts[3] {1, 1, 1};
  // decide how to cut up mesh
  {
    IdxT P = comm_size;
    IdxT sqrtP = sqrt(P);
    
    // get prime factors of P
    std::multiset<IdxT> prime_factors;
    {
      // get list of possible prime factors (excluding P)
      std::vector<IdxT> primes({2});
      for(IdxT p = 3; p < sqrtP; p += 2) {
        IdxT primes_size = primes.size();
        IdxT i = 0;
        for (; i < primes_size; ++i) {
          if (p % primes[i] == 0) break; // not prime
        }
        if (i == primes_size) {
          primes.push_back(p);
        }
      }
    
      IdxT P_tmp = P;
      IdxT pi = 0;
      IdxT primes_size = primes.size();
      while(pi != primes_size) {
        if (P_tmp % primes[pi] == 0) {
          // found prime factor
          prime_factors.insert(primes[pi]);
          P_tmp /= primes[pi];
        } else {
          ++pi;
        }
      }
      if (P_tmp != 1) {
        prime_factors.insert(P_tmp);
      }
    }
    
    double I = std::cbrt(sizes[0] * sizes[1] * sizes[2]) / P;
    
    while(std::begin(prime_factors) != std::end(prime_factors)) {
      
      double best_relative_remainder_dist = 1e100;
      auto best_factor = std::end(prime_factors);
      IdxT best_dim = 3;
      
      double ideal[3] {sizes[0] / (cuts[0] * I),
                       sizes[1] / (cuts[1] * I),
                       sizes[2] / (cuts[2] * I)};
      
      auto end = std::end(prime_factors);
      for(auto k = std::begin(prime_factors); k != end; ++k) {
        IdxT factor = *k;
        for(IdxT dim = 2; dim >= 0; --dim) {
          double remainder = std::fmod(ideal[dim], factor);
          double remainder_dist = std::min(remainder, factor - remainder);
          double relative_remainder_dist = remainder_dist / factor;
          if (relative_remainder_dist < best_relative_remainder_dist) {
            best_relative_remainder_dist = relative_remainder_dist;
            best_factor = k;
            best_dim = dim;
          }
        }
      }
      
      assert(best_factor != end && best_dim < 3);
      
      cuts[best_dim] *= *best_factor;
      
      prime_factors.erase(best_factor);
    }
  }
  
  // create cartesian communicator
  MPI_Comm cart_comm;
  
  MPI_Cart_create(mpi_comm, 3, cuts, periodic, 1, &cart_comm);
  
  int cart_rank = -1;
  MPI_Comm_rank(cart_comm, &cart_rank);
  
  int cart_coords[3] {-1, -1, -1};
  MPI_Cart_coords(cart_comm, cart_rank, 3, cart_coords);
  
  IdxT local_min[3] { (cart_coords[0] * sizes[0]) / cuts[0]
                    , (cart_coords[1] * sizes[1]) / cuts[1]
                    , (cart_coords[2] * sizes[2]) / cuts[2] };
                    
  IdxT local_max[3] { ((cart_coords[0]+1) * sizes[0]) / cuts[0]
                    , ((cart_coords[1]+1) * sizes[1]) / cuts[1]
                    , ((cart_coords[2]+1) * sizes[2]) / cuts[2] };
    
  IdxT local_sizes[3] { local_max[0] - local_min[0]
                      , local_max[1] - local_min[1]
                      , local_max[2] - local_min[2] };
  
  
  MeshInfo global_info(sizes[0], sizes[1], sizes[2], ghost_width);
  MeshInfo info(local_sizes[0], local_sizes[1], local_sizes[2], ghost_width);
    
  if (comm_rank == 0) {
    printf("Num cycles %i\n", ncycles);
    printf("Num vars %i\n", num_vars);
    printf("Global Mesh info\n");
    printf("%i %i %i\n", global_info.isize, global_info.jsize, global_info.ksize);
    printf("Local Mesh info\n");
    printf("%i %i %i\n", info.isize, info.jsize, info.ksize);
    printf("ij %i ik %i jk %i\n", info.ijsize, info.iksize, info.jksize);
    printf("ijk %i\n", info.ijksize);
    printf("ghost_width %i\n", info.ghost_width);
    printf("i %8i %8i %8i %8i\n", 0, info.imin, info.imax, info.ilen);
    printf("j %8i %8i %8i %8i\n", 0, info.jmin, info.jmax, info.jlen);
    printf("k %8i %8i %8i %8i\n", 0, info.kmin, info.kmax, info.klen);
    printf("ij %i ik %i jk %i\n", info.ijlen, info.iklen, info.jklen);
    printf("ijk %i\n", info.ijklen);
    fflush(stdout);
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
    
    printf("Starting up memory pools\n"); fflush(stdout);

    DataT** vars = new DataT*[num_vars+1];
 
    tm.start(host_alloc.name());

    for (IdxT i = 0; i < num_vars+1; ++i) {
      vars[i] = (DataT*)host_alloc.allocate(info.ijklen*sizeof(DataT));
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      IdxT len = info.ijklen;
      DataT* data = vars[i];
      for_all(seq_pol{}, 0, len, [=](IdxT, IdxT idx) {
        data[idx] = 0.0;
      });
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      host_alloc.deallocate(vars[i]);
    }

    tm.restart(hostpinned_alloc.name());

    for (IdxT i = 0; i < num_vars+1; ++i) {
      vars[i] = (DataT*)hostpinned_alloc.allocate(info.ijklen*sizeof(DataT));
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      IdxT len = info.ijklen;
      DataT* data = vars[i];
      for_all(seq_pol{}, 0, len, [=](IdxT, IdxT idx) {
        data[idx] = 0.0;
      });
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      hostpinned_alloc.deallocate(vars[i]);
    }

    tm.restart(device_alloc.name());

    for (IdxT i = 0; i < num_vars+1; ++i) {
      vars[i] = (DataT*)device_alloc.allocate(info.ijklen*sizeof(DataT));
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      IdxT len = info.ijklen;
      DataT* data = vars[i];
      for_all(cuda_pol{}, 0, len, [=] DEVICE (IdxT, IdxT idx) {
        data[idx] = 0.0;
      });
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      device_alloc.deallocate(vars[i]);
    }

    tm.restart(managed_alloc.name());

    for (IdxT i = 0; i < num_vars+1; ++i) {
      vars[i] = (DataT*)managed_alloc.allocate(info.ijklen*sizeof(DataT));
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      IdxT len = info.ijklen;
      DataT* data = vars[i];
      for_all(cuda_pol{}, 0, len, [=] DEVICE (IdxT, IdxT idx) {
        data[idx] = 0.0;
      });
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      managed_alloc.deallocate(vars[i]);
    }

    tm.restart(managed_host_preferred_alloc.name());

    for (IdxT i = 0; i < num_vars+1; ++i) {
      vars[i] = (DataT*)managed_host_preferred_alloc.allocate(info.ijklen*sizeof(DataT));
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      IdxT len = info.ijklen;
      DataT* data = vars[i];
      for_all(seq_pol{}, 0, len, [=](IdxT, IdxT idx) {
        data[idx] = 0.0;
      });
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      managed_host_preferred_alloc.deallocate(vars[i]);
    }

    tm.restart(managed_host_preferred_device_accessed_alloc.name());

    for (IdxT i = 0; i < num_vars+1; ++i) {
      vars[i] = (DataT*)managed_host_preferred_device_accessed_alloc.allocate(info.ijklen*sizeof(DataT));
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      IdxT len = info.ijklen;
      DataT* data = vars[i];
      for_all(seq_pol{}, 0, len, [=](IdxT, IdxT idx) {
        data[idx] = 0.0;
      });
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      managed_host_preferred_device_accessed_alloc.deallocate(vars[i]);
    }

    tm.restart(managed_device_preferred_alloc.name());

    for (IdxT i = 0; i < num_vars+1; ++i) {
      vars[i] = (DataT*)managed_device_preferred_alloc.allocate(info.ijklen*sizeof(DataT));
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      IdxT len = info.ijklen;
      DataT* data = vars[i];
      for_all(cuda_pol{}, 0, len, [=] DEVICE (IdxT, IdxT idx) {
        data[idx] = 0.0;
      });
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      managed_device_preferred_alloc.deallocate(vars[i]);
    }

    tm.restart(managed_device_preferred_host_accessed_alloc.name());

    for (IdxT i = 0; i < num_vars+1; ++i) {
      vars[i] = (DataT*)managed_device_preferred_host_accessed_alloc.allocate(info.ijklen*sizeof(DataT));
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      IdxT len = info.ijklen;
      DataT* data = vars[i];
      for_all(cuda_pol{}, 0, len, [=] DEVICE (IdxT, IdxT idx) {
        data[idx] = 0.0;
      });
    }
    for (IdxT i = 0; i < num_vars+1; ++i) {
      managed_device_preferred_host_accessed_alloc.deallocate(vars[i]);
    }

    tm.stop();
    
    delete[] vars;

    tm.print();
    tm.clear();

  }

  // host allocated
  {
    Allocator& mesh_aloc = host_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);
  }

  // host pinned allocated
  {
    Allocator& mesh_aloc = hostpinned_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // device allocated
  {
    Allocator& mesh_aloc = device_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    // do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    // do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
 
    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, device_alloc, device_alloc, device_alloc, tm);
  }

  // managed allocated
  {
    Allocator& mesh_aloc = managed_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed host preferred allocated
  {
    Allocator& mesh_aloc = managed_host_preferred_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed host preferred device accessed allocated
  {
    Allocator& mesh_aloc = managed_host_preferred_device_accessed_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed device preferred allocated
  {
    Allocator& mesh_aloc = managed_device_preferred_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  // managed device preferred host accessed allocated
  {
    Allocator& mesh_aloc = managed_device_preferred_host_accessed_alloc;

    char name[1024] = ""; snprintf(name, 1024, "Mesh %s", mesh_aloc.name());
    Range r0(name, Range::blue);

    do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, host_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, seq_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, host_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, seq_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, host_alloc, tm);

    do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>(info, num_vars, ncycles, mesh_aloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);
  }

  MPI_Finalize();
  return 0;
}

