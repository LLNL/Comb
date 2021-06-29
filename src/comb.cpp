//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2021, Lawrence Livermore National Security, LLC.
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
#include "CommFactory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cctype>
#include <unistd.h>
#include <sched.h>

#define PRINT_THREAD_MAP

#ifdef PRINT_THREAD_MAP
#include <linux/sched.h>
#endif

#ifdef COMB_ENABLE_CALIPER
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#endif
#ifdef COMB_ENABLE_ADIAK
#include <adiak.hpp>
#endif

int main(int argc, char** argv)
{
  void* adiak_comm_p = nullptr;

#ifdef COMB_ENABLE_MPI
  int required = MPI_THREAD_FUNNELED; // MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE
  int provided = detail::MPI::Init_thread(&argc, &argv, required);

  MPI_Comm adiak_comm = detail::MPI::Comm_dup(MPI_COMM_WORLD);
  adiak_comm_p = &adiak_comm;
#endif

#ifdef COMB_ENABLE_ADIAK
  adiak_init(adiak_comm_p);
#endif

  comb_setup_files();

  fgprintf(FileGroup::all, "Comb version %i.%i.%i\n",
      COMB_VERSION_MAJOR, COMB_VERSION_MINOR, COMB_VERSION_PATCHLEVEL);

  fgprintf(FileGroup::all, "Args  %s", argv[0]);
  for(int i = 1; i < argc; ++i) {
    fgprintf(FileGroup::all, ";%s", argv[i]);
  }
  fgprintf(FileGroup::all, "\n");

  { // begin region MPI communication via comminfo
  CommInfo comminfo;

#ifdef COMB_ENABLE_MPI
  if (required != provided) {
    fgprintf(FileGroup::err_master, "Didn't receive MPI thread support required %i provided %i.\n", required, provided);
    comminfo.abort();
  }
#endif

  fgprintf(FileGroup::all, "Started rank %i of %i\n", comminfo.rank, comminfo.size);

  {
    char host[256];
    gethostname(host, 256);

    fgprintf(FileGroup::all, "Node %s\n", host);
  }

  fgprintf(FileGroup::all, "Compiler %s\n", COMB_SERIALIZE(COMB_COMPILER));

#ifdef COMB_ENABLE_CUDA
  {
    fgprintf(FileGroup::all, "Cuda compiler %s\n", COMB_SERIALIZE(COMB_CUDA_COMPILER));

    int driver_v = -1;
    cudaCheck(cudaDriverGetVersion(&driver_v));
    fgprintf(FileGroup::all, "Cuda driver version %i\n", driver_v);

    int runtime_v = -1;
    cudaCheck(cudaRuntimeGetVersion(&runtime_v));
    fgprintf(FileGroup::all, "Cuda runtime version %i\n", runtime_v);

    const char* visible_devices = nullptr;
    visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    if (visible_devices == nullptr) {
      visible_devices = "undefined";
    }

    int device = -1;
    cudaCheck(cudaGetDevice(&device));

    fgprintf(FileGroup::all, "GPU %i visible %s\n", device, visible_devices);

    cudaCheck(cudaDeviceSynchronize());
  }
#endif


  // stores the Allocator for each memory type,
  // whether each memory type is available for use,
  // and whether each memory type is accessbile from each exec context
  COMB::Allocators alloc;

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

  bool do_basic_only = false;

  bool do_print_packing_sizes = false;
  bool do_print_message_sizes = false;

  // Caliper profiling config, if enabled
  std::string caliper_config;

  // stores whether each comm policy is available for use
  COMB::CommunicatorsAvailable comm_avail;
  comm_avail.mock = true;
#ifdef COMB_ENABLE_MPI
  comm_avail.mpi = true;
#endif

  // stores whether each exec policy is available for use
  COMB::Executors exec;
  exec.seq.m_available = true;

  // stores whether each memory type is available for use
  alloc.host.m_available = true;

  IdxT i = 1;
  IdxT s = 0;
  for(; i < argc; ++i) {
    if (argv[i][0] == '-') {
      // options
      if (strcmp(&argv[i][1], "comm") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          ++i;
          if (strcmp(argv[i], "cutoff") == 0) {
            if (i+1 < argc && argv[i+1][0] != '-') {
              long read_cutoff = comminfo.cutoff;
              int ret = sscanf(argv[++i], "%ld", &read_cutoff);
              if (ret == 1) {
                comminfo.cutoff = read_cutoff;
              } else {
                fgprintf(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              fgprintf(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
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
                fgprintf(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              fgprintf(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
            }
          } else if ( strcmp(argv[i], "enable") == 0
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
                comm_avail.mock = enabledisable;
#ifdef COMB_ENABLE_MPI
                comm_avail.mpi = enabledisable;
#endif
#ifdef COMB_ENABLE_MPI
                comm_avail.mpi_persistent = enabledisable;
#endif
#ifdef COMB_ENABLE_GDSYNC
                comm_avail.gdsync = enabledisable;
#endif
#ifdef COMB_ENABLE_GPUMP
                comm_avail.gpump = enabledisable;
#endif
#ifdef COMB_ENABLE_MP
                comm_avail.mp = enabledisable;
#endif
#ifdef COMB_ENABLE_UMR
                comm_avail.umr = enabledisable;
#endif
              } else if (strcmp(argv[i], "mock") == 0) {
                comm_avail.mock = enabledisable;
              } else if (strcmp(argv[i], "mpi") == 0) {
#ifdef COMB_ENABLE_MPI
                comm_avail.mpi = enabledisable;
#endif
              } else if (strcmp(argv[i], "mpi_persistent") == 0) {
#ifdef COMB_ENABLE_MPI
                comm_avail.mpi_persistent = enabledisable;
#endif
              } else if (strcmp(argv[i], "gdsync") == 0) {
#ifdef COMB_ENABLE_GDSYNC
                comm_avail.gdsync = enabledisable;
#endif
              } else if (strcmp(argv[i], "gpump") == 0) {
#ifdef COMB_ENABLE_GPUMP
                comm_avail.gpump = enabledisable;
#endif
              } else if (strcmp(argv[i], "mp") == 0) {
#ifdef COMB_ENABLE_MP
                comm_avail.mp = enabledisable;
#endif
              } else if (strcmp(argv[i], "umr") == 0) {
#ifdef COMB_ENABLE_UMR
                comm_avail.umr = enabledisable;
#endif
              } else {
                fgprintf(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              fgprintf(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
            }
          } else if ( strcmp(argv[i], "allow") == 0
                   || strcmp(argv[i], "disallow") == 0 ) {
            bool allowdisallow = false;
            if (strcmp(argv[i], "allow") == 0) {
              allowdisallow = true;
            } else if (strcmp(argv[i], "disallow") == 0) {
              allowdisallow = false;
            }
            if (i+1 < argc && argv[i+1][0] != '-') {
              ++i;
              if (strcmp(argv[i], "per_message_pack_fusing") == 0) {
                comb_allow_per_message_pack_fusing() = allowdisallow;
              } else if (strcmp(argv[i], "message_group_pack_fusing") == 0) {
                comb_allow_pack_loop_fusion() = allowdisallow;
              } else {
                fgprintf(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              fgprintf(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
            }
          } else {
            fgprintf(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
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
            fgprintf(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
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
                exec.seq.m_available = enabledisable;
#ifdef COMB_ENABLE_OPENMP
                exec.omp.m_available = enabledisable;
#endif
#ifdef COMB_ENABLE_CUDA
                exec.cuda.m_available = enabledisable;
#endif
#ifdef COMB_ENABLE_CUDA_GRAPH
                exec.cuda_graph.m_available = enabledisable;
#endif
#ifdef COMB_ENABLE_MPI
                exec.mpi_type.m_available = enabledisable;
#endif
#ifdef COMB_ENABLE_RAJA
                exec.raja_seq.m_available = enabledisable;
#ifdef COMB_ENABLE_OPENMP
                exec.raja_omp.m_available = enabledisable;
#endif
#ifdef COMB_ENABLE_CUDA
                exec.raja_cuda.m_available = enabledisable;
#endif
#endif
              } else if (strcmp(argv[i], "seq") == 0) {
                exec.seq.m_available = enabledisable;
              } else if (strcmp(argv[i], "omp") == 0 ||
                         strcmp(argv[i], "openmp") == 0) {
#ifdef COMB_ENABLE_OPENMP
                exec.omp.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "cuda") == 0) {
#ifdef COMB_ENABLE_CUDA
                exec.cuda.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "cuda_graph") == 0) {
#ifdef COMB_ENABLE_CUDA_GRAPH
                exec.cuda_graph.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "mpi_type") == 0) {
#ifdef COMB_ENABLE_MPI
                exec.mpi_type.m_available = enabledisable;
#endif
#ifdef COMB_ENABLE_RAJA
              } else if (strcmp(argv[i], "raja_seq") == 0) {
                exec.raja_seq.m_available = enabledisable;
#ifdef COMB_ENABLE_OPENMP
              } else if (strcmp(argv[i], "raja_omp") == 0 ||
                         strcmp(argv[i], "raja_openmp") == 0) {
                exec.raja_omp.m_available = enabledisable;
#endif
#ifdef COMB_ENABLE_CUDA
              } else if (strcmp(argv[i], "raja_cuda") == 0) {
                exec.raja_cuda.m_available = enabledisable;
#endif
#endif
              } else {
                fgprintf(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              fgprintf(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
            }
          } else {
            fgprintf(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
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
                alloc.host.m_available = enabledisable;
#ifdef COMB_ENABLE_CUDA
                alloc.cuda_hostpinned.m_available = enabledisable;
                alloc.cuda_device.m_available = enabledisable;
                alloc.cuda_managed.m_available = enabledisable;
                alloc.cuda_managed_host_preferred.m_available = enabledisable;
                alloc.cuda_managed_host_preferred_device_accessed.m_available = enabledisable;
                alloc.cuda_managed_device_preferred.m_available = enabledisable;
                alloc.cuda_managed_device_preferred_host_accessed.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "host") == 0) {
                alloc.host.m_available = enabledisable;
              } else if (strcmp(argv[i], "cuda_hostpinned") == 0) {
#ifdef COMB_ENABLE_CUDA
                alloc.cuda_hostpinned.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "cuda_device") == 0) {
#ifdef COMB_ENABLE_CUDA
                alloc.cuda_device.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "cuda_managed") == 0) {
#ifdef COMB_ENABLE_CUDA
                alloc.cuda_managed.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "cuda_managed_host_preferred") == 0) {
#ifdef COMB_ENABLE_CUDA
                alloc.cuda_managed_host_preferred.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "cuda_managed_host_preferred_device_accessed") == 0) {
#ifdef COMB_ENABLE_CUDA
                alloc.cuda_managed_host_preferred_device_accessed.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "cuda_managed_device_preferred") == 0) {
#ifdef COMB_ENABLE_CUDA
                alloc.cuda_managed_device_preferred.m_available = enabledisable;
#endif
              } else if (strcmp(argv[i], "cuda_managed_device_preferred_host_accessed") == 0) {
#ifdef COMB_ENABLE_CUDA
                alloc.cuda_managed_device_preferred_host_accessed.m_available = enabledisable;
#endif
              } else {
                fgprintf(FileGroup::err_master, "Invalid argument to sub-option, ignoring %s %s %s.\n", argv[i-2], argv[i-1], argv[i]);
              }
            } else {
              fgprintf(FileGroup::err_master, "No argument to sub-option, ignoring %s %s.\n", argv[i-1], argv[i]);
            }
          } else {
            fgprintf(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "vars") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          long read_num_vars = num_vars;
          int ret = sscanf(argv[++i], "%ld", &read_num_vars);
          if (ret == 1) {
            num_vars = read_num_vars;
          } else {
            fgprintf(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "cycles") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          long read_ncycles = ncycles;
          int ret = sscanf(argv[++i], "%ld", &read_ncycles);
          if (ret == 1) {
            ncycles = read_ncycles;
          } else {
            fgprintf(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
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
            fgprintf(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
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
            fgprintf(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
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
            fgprintf(FileGroup::err_master, "Not built with openmp, ignoring %s %s.\n", argv[i-1], argv[i]);
#endif
          } else {
            fgprintf(FileGroup::err_master, "Invalid argument to option, ignoring %s %s.\n", argv[i-1], argv[i]);
          }
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else if (strcmp(&argv[i][1], "basic_only") == 0) {
        do_basic_only = true;
      } else if (strcmp(&argv[i][1], "cuda_aware_mpi") == 0) {
#ifdef COMB_ENABLE_MPI
#ifdef COMB_ENABLE_CUDA
        alloc.access.cuda_aware_mpi = true;
#else
        fgprintf(FileGroup::err_master, "Not built with cuda, ignoring %s.\n", argv[i]);
#endif
#else
        fgprintf(FileGroup::err_master, "Not built with mpi, ignoring %s.\n", argv[i]);
#endif
      } else if (strcmp(&argv[i][1], "cuda_host_accessible_from_device") == 0) {
#ifdef COMB_ENABLE_CUDA
        alloc.access.cuda_host_accessible_from_device = COMB::detail::cuda::get_host_accessible_from_device();
#else
        fgprintf(FileGroup::err_master, "Not built with cuda, ignoring %s.\n", argv[i]);
#endif
      } else if (strcmp(&argv[i][1], "cuda_device_accessible_from_host") == 0) {
#ifdef COMB_ENABLE_CUDA
        alloc.access.cuda_device_accessible_from_host = COMB::detail::cuda::get_device_accessible_from_host();
#else
        fgprintf(FileGroup::err_master, "Not built with cuda, ignoring %s.\n", argv[i]);
#endif
      } else if (strcmp(&argv[i][1], "use_device_preferred_for_cuda_util_aloc") == 0) {
#ifdef COMB_ENABLE_CUDA
        alloc.access.use_device_preferred_for_cuda_util_aloc = true;
#else
        fgprintf(FileGroup::err_master, "Not built with cuda, ignoring %s.\n", argv[i]);
#endif
      } else if (strcmp(&argv[i][1], "print_packing_sizes") == 0) {
        do_print_packing_sizes = true;
      } else if (strcmp(&argv[i][1], "print_message_sizes") == 0) {
        do_print_message_sizes = true;
      } else if (strcmp(&argv[i][1], "caliper_config") == 0) {
        if (i+1 < argc && argv[i+1][0] != '-') {
          caliper_config = argv[++i];
#ifndef COMB_ENABLE_CALIPER
          fgprintf(FileGroup::err_master, "Caliper is not enabled, ignoring caliper_config.\n");
#endif
        } else {
          fgprintf(FileGroup::err_master, "No argument to option, ignoring %s.\n", argv[i]);
        }
      } else {
        fgprintf(FileGroup::err_master, "Unknown option, ignoring %s.\n", argv[i]);
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
        fgprintf(FileGroup::err_master, "Invalid argument to sizes, ignoring %s.\n", argv[i]);
      }
    } else {
      fgprintf(FileGroup::err_master, "Invalid argument, ignoring %s.\n", argv[i]);
    }
  }

  if (ncycles <= 0) {
    fgprintf(FileGroup::err_master, "Invalid cycles argument.\n");
    comminfo.abort();
  } else if (num_vars <= 0) {
    fgprintf(FileGroup::err_master, "Invalid vars argument.\n");
    comminfo.abort();
  } else if ( (ghost_widths[0] <  0 || ghost_widths[1] <  0 || ghost_widths[2] <  0)
           || (ghost_widths[0] == 0 && ghost_widths[1] == 0 && ghost_widths[2] == 0) ) {
    fgprintf(FileGroup::err_master, "Invalid ghost widths.\n");
    comminfo.abort();
  } else if ( (divisions[0] != 0 || divisions[1] != 0 || divisions[2] != 0)
           && (comminfo.size != divisions[0] * divisions[1] * divisions[2]) ) {
    fgprintf(FileGroup::err_master, "Invalid mesh divisions\n");
    comminfo.abort();
  }

#ifdef COMB_ENABLE_CALIPER
  cali::ConfigManager mgr;
  mgr.add(caliper_config.c_str());

  if (mgr.error()) {
    std::string msg = mgr.error_msg();
    fgprintf(FileGroup::err_master, "Caliper config error: %s", msg.c_str());
    comminfo.abort();
  }

  mgr.start();

  CALI_MARK_FUNCTION_BEGIN;
#endif

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
    fgprintf(FileGroup::all, "OMP num threads %5li\n", print_omp_threads);

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
        fgprintf(FileGroup::all, "OMP thread map %6i", thread_cpu_id[i]);
        for (++i; i < omp_threads; ++i) {
          fgprintf(FileGroup::all, ";%i", thread_cpu_id[i]);
        }
      }

      fgprintf(FileGroup::all, "\n");

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

    fgprintf(FileGroup::all, "Cart coords  %8li %8li %8li\n", print_coords[0],       print_coords[1],       print_coords[2]      );
    fgprintf(FileGroup::all, "Message policy cutoff %li\n",   print_cutoff                                                       );
    fgprintf(FileGroup::all, "Post Recv using %s method\n",   CommInfo::method_str(comminfo.post_recv_method)                    );
    fgprintf(FileGroup::all, "Post Send using %s method\n",   CommInfo::method_str(comminfo.post_send_method)                    );
    fgprintf(FileGroup::all, "Wait Recv using %s method\n",   CommInfo::method_str(comminfo.wait_recv_method)                    );
    fgprintf(FileGroup::all, "Wait Send using %s method\n",   CommInfo::method_str(comminfo.wait_send_method)                    );
    fgprintf(FileGroup::all, "Num cycles   %8li\n",           print_ncycles                                                      );
    fgprintf(FileGroup::all, "Num vars     %8li\n",           print_num_vars                                                     );
    fgprintf(FileGroup::all, "ghost_widths %8li %8li %8li\n", print_ghost_widths[0], print_ghost_widths[1], print_ghost_widths[2]);
    fgprintf(FileGroup::all, "sizes        %8li %8li %8li\n", print_sizes[0],        print_sizes[1],        print_sizes[2]       );
    fgprintf(FileGroup::all, "divisions    %8li %8li %8li\n", print_divisions[0],    print_divisions[1],    print_divisions[2]   );
    fgprintf(FileGroup::all, "periodic     %8li %8li %8li\n", print_periodic[0],     print_periodic[1],     print_periodic[2]    );
    fgprintf(FileGroup::all, "division map\n");
    // print division map
    IdxT max_cuts = std::max(std::max(comminfo.cart.divisions[0], comminfo.cart.divisions[1]), comminfo.cart.divisions[2]);
    for (IdxT ci = 0; ci <= max_cuts; ++ci) {
      fgprintf(FileGroup::all, "map         ");
      if (ci <= comminfo.cart.divisions[0]) {
        long print_division_coord = ci * (sizes[0] / comminfo.cart.divisions[0]) + std::min(ci, sizes[0] % comminfo.cart.divisions[0]);
        fgprintf(FileGroup::all, " %8li", print_division_coord);
      } else {
        fgprintf(FileGroup::all, " %8s", "");
      }
      if (ci <= comminfo.cart.divisions[1]) {
        long print_division_coord = ci * (sizes[1] / comminfo.cart.divisions[1]) + std::min(ci, sizes[1] % comminfo.cart.divisions[1]);
        fgprintf(FileGroup::all, " %8li", print_division_coord);
      } else {
        fgprintf(FileGroup::all, " %8s", "");
      }
      if (ci <= comminfo.cart.divisions[2]) {
        long print_division_coord = ci * (sizes[2] / comminfo.cart.divisions[2]) + std::min(ci, sizes[2] % comminfo.cart.divisions[2]);
        fgprintf(FileGroup::all, " %8li", print_division_coord);
      } else {
        fgprintf(FileGroup::all, " %8s", "");
      }
      fgprintf(FileGroup::all, "\n");
    }

    // save config info in Adiak
#ifdef COMB_ENABLE_ADIAK
    adiak_namevalue("cart_coords",   adiak_general, nullptr, "[%ld]", print_coords,       3);
    adiak_namevalue("ghost_width",   adiak_general, nullptr, "[%ld]", print_ghost_widths, 3);
    adiak_namevalue("sizes",         adiak_general, nullptr, "[%ld]", print_sizes,        3);
    adiak_namevalue("divisions",     adiak_general, nullptr, "[%ld]", print_divisions,    3);
    adiak_namevalue("periodic",      adiak_general, nullptr, "[%ld]", print_periodic,     3);

    adiak::value("policy_cutoff",    print_cutoff);
    adiak::value("ncycles",          print_ncycles);
    adiak::value("num_vars",         print_num_vars);
    adiak::value("post_recv_method", CommInfo::method_str(comminfo.post_recv_method));
    adiak::value("post_send_method", CommInfo::method_str(comminfo.post_send_method));
    adiak::value("wait_recv_method", CommInfo::method_str(comminfo.wait_recv_method));
    adiak::value("wait_send_method", CommInfo::method_str(comminfo.wait_send_method));

    adiak_user();
    adiak_launchdate();
    adiak_cmdline();
    adiak_clustername();
    adiak_job_size();
#endif
  }

  COMB::print_message_info(comminfo, info, alloc.host.allocator(), num_vars, do_print_packing_sizes, do_print_message_sizes);

  Timer tm(2*6*ncycles);
  Timer tm_total(1024);

  exec.create_executors(alloc);

  // warm-up memory pools
  COMB::warmup(exec, alloc, tm, num_vars+1, info.totallen);

  COMB::test_copy(comminfo, exec, alloc, tm, num_vars, info.totallen, ncycles);

  if (do_basic_only) {

    COMB::test_cycles_basic(comminfo, info, exec, alloc, num_vars, ncycles, tm, tm_total);

  } else {

    if (comm_avail.mock)
      COMB::test_cycles_mock(comminfo, info, exec, alloc, num_vars, ncycles, tm, tm_total);

#ifdef COMB_ENABLE_MPI
    if (comm_avail.mpi)
      COMB::test_cycles_mpi(comminfo, info, exec, alloc, num_vars, ncycles, tm, tm_total);
#endif

#ifdef COMB_ENABLE_MPI
    if (comm_avail.mpi_persistent)
      COMB::test_cycles_mpi_persistent(comminfo, info, exec, alloc, num_vars, ncycles, tm, tm_total);
#endif

#ifdef COMB_ENABLE_GDSYNC
    if (comm_avail.gdsync)
      COMB::test_cycles_gdsync(comminfo, info, exec, alloc, num_vars, ncycles, tm, tm_total);
#endif

#ifdef COMB_ENABLE_GPUMP
    if (comm_avail.gpump)
      COMB::test_cycles_gpump(comminfo, info, exec, alloc, num_vars, ncycles, tm, tm_total);
#endif

#ifdef COMB_ENABLE_MP
    if (comm_avail.mp)
      COMB::test_cycles_mp(comminfo, info, exec, alloc, num_vars, ncycles, tm, tm_total);
#endif

#ifdef COMB_ENABLE_UMR
    if (comm_avail.umr)
      COMB::test_cycles_umr(comminfo, info, exec, alloc, num_vars, ncycles, tm, tm_total);
#endif

  }

#ifdef COMB_ENABLE_CALIPER
  CALI_MARK_FUNCTION_END;
  mgr.flush();
#endif

  } // end region MPI communication via comminfo

  comb_teardown_files();

#ifdef COMB_ENABLE_ADIAK
  adiak_walltime();
  adiak_fini();
#endif

#ifdef COMB_ENABLE_MPI
  detail::MPI::Finalize();
#endif
  return 0;
}

