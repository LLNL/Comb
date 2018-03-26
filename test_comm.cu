#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <chrono>

#include <cuda.h>
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
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

template < typename pol_loop, typename pol_corner, typename pol_edge, typename pol_face >
void do_cycles(const char* name, MeshInfo& mesh, IdxT ncycles, Allocator& aloc_mesh, Allocator& aloc_corner, Allocator& aloc_edge, Allocator& aloc_face, Timer& tm)
{
    tm.clear();

    printf("Starting test %s\n", name); fflush(stdout);

    Range r0(name, Range::orange);

    tm.start("start-up");

    MeshData var(mesh, aloc_mesh);
    Comm<pol_corner, pol_edge, pol_face> comm(var, aloc_corner, aloc_edge, aloc_face);

    {
      var.allocate();
      DataT* data = var.data();
      IdxT ijklen = mesh.ijklen;

      for_all(pol_loop{}, 0, ijklen,
                          detail::set_n1(data));

      if (pol_loop::async) {cudaCheck(cudaDeviceSynchronize());}

    }

    tm.stop();

    for(IdxT cycle = 0; cycle < ncycles; cycle++) {

      Range r1("cycle", Range::yellow);

      IdxT imin = mesh.imin;
      IdxT jmin = mesh.jmin;
      IdxT kmin = mesh.kmin;
      IdxT imax = mesh.imax;
      IdxT jmax = mesh.jmax;
      IdxT kmax = mesh.kmax;
      IdxT ilen = mesh.ilen;
      IdxT jlen = mesh.jlen;
      IdxT klen = mesh.klen;
      IdxT ijlen = mesh.ijlen;
      
      DataT* data = var.data();
      
      Range r2("pre-comm", Range::red);
      tm.start("pre-comm");

      for_all_3d(pol_loop{}, kmin, kmax,
                             jmin, jmax,
                             imin, imax,
                             detail::set_1(ilen, ijlen, data));

      if (pol_loop::async) {cudaCheck(cudaDeviceSynchronize());}

      tm.stop();
      r2.restart("post-recv", Range::pink);
      tm.start("post-recv");
      
      comm.postRecv();

      tm.stop();
      r2.restart("post-send", Range::pink);
      tm.start("post-send");

      comm.postSend();

      if (pol_corner::async || pol_edge::async || pol_face::async) {cudaCheck(cudaDeviceSynchronize());}

/*      for_all_3d(pol_loop{}, 0, klen,
                            0, jlen,
                            0, ilen,
                            [=] (IdxT k, IdxT j, IdxT i, IdxT idx) {
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
        //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
        data[zone] = next;
      });
*/

      tm.stop();
      r2.restart("wait-send", Range::pink);
      tm.start("wait-send");

      comm.waitSend();

      if (pol_corner::async || pol_edge::async || pol_face::async) {cudaCheck(cudaDeviceSynchronize());}

      tm.stop();
      r2.restart("wait-recv", Range::pink);
      tm.start("wait-recv");

      comm.waitRecv();

      tm.stop();
      r2.restart("post-comm", Range::red);
      tm.start("post-comm");

      for_all_3d(pol_loop{}, 0, klen,
                             0, jlen,
                             0, ilen,
                             detail::reset_1(ilen, ijlen, data, imin, jmin, kmin, imax, jmax, kmax));

      if (pol_loop::async) {cudaCheck(cudaDeviceSynchronize());}

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

  IdxT isize = 0;
  IdxT jsize = 0;
  IdxT ksize = 0;

  if (argc == 1) {
    isize = 100;
    jsize = 100;
    ksize = 100;
  } else if (argc == 2) {
    isize = static_cast<IdxT>(atoll(argv[1]));
    jsize = isize;
    ksize = isize;
  } else if (argc == 4) {
    isize = static_cast<IdxT>(atoll(argv[1]));
    jsize = static_cast<IdxT>(atoll(argv[2]));
    ksize = static_cast<IdxT>(atoll(argv[3]));
  } else {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid arguments.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (isize <= 0 || jsize <= 0 || ksize <= 0) {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid size arguments.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MeshInfo mesh(isize, jsize, ksize);
    
  if (comm_rank == 0) {
    printf("Mesh info\n");
    printf("%i %i %i\n", mesh.isize, mesh.jsize, mesh.ksize);
    printf("ij %i ik %i jk %i\n", mesh.ijsize, mesh.iksize, mesh.jksize);
    printf("ijk %i\n", mesh.ijksize);
    printf("i %8i %8i %8i %8i\n", 0, mesh.imin, mesh.imax, mesh.ilen);
    printf("j %8i %8i %8i %8i\n", 0, mesh.jmin, mesh.jmax, mesh.jlen);
    printf("k %8i %8i %8i %8i\n", 0, mesh.kmin, mesh.kmax, mesh.klen);
    printf("ij %i ik %i jk %i\n", mesh.ijlen, mesh.iklen, mesh.jklen);
    printf("ijk %i\n", mesh.ijklen);
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
    printf("Starting up memory pools\n"); fflush(stdout);

    Range r("Memmory pool init", Range::green);

    void* var0;
    void* var1;
 
    tm.start("host");

    var0 = host_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = host_alloc.allocate(mesh.ijksize*sizeof(DataT));
    host_alloc.deallocate(var0);
    host_alloc.deallocate(var1);

    tm.restart("host_pinned");

    var0 = hostpinned_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = hostpinned_alloc.allocate(mesh.ijksize*sizeof(DataT));
    hostpinned_alloc.deallocate(var0);
    hostpinned_alloc.deallocate(var1);

    tm.restart("device");

    var0 = device_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = device_alloc.allocate(mesh.ijksize*sizeof(DataT));
    device_alloc.deallocate(var0);
    device_alloc.deallocate(var1);

    tm.restart("managed");

    var0 = managed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = managed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    managed_alloc.deallocate(var0);
    managed_alloc.deallocate(var1);

    tm.stop();

    tm.print();
    tm.clear();

  }

  IdxT ncycles = 5;

  // host allocated
  do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>("Host_seq", mesh, ncycles, host_alloc, host_alloc, host_alloc, host_alloc, tm);
 
  // host pinned allocated
  do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>("hostpinned_seq", mesh, ncycles, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);

  do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>("hostpinned_cuda_seq", mesh, ncycles, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);

  do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>("hostpinned_cuda", mesh, ncycles, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);

  // device allocated
  do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>("device_hostpinned_cuda", mesh, ncycles, device_alloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);

  do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>("device_cuda", mesh, ncycles, device_alloc, device_alloc, device_alloc, device_alloc, tm);

  MPI_Finalize();
  return 0;
}

