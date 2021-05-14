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

#ifndef _UTILS_GPUMP_HPP
#define _UTILS_GPUMP_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_GPUMP

#include <libgpump.h>

#include <cassert>
#include <cstdio>

#include "exec_utils.hpp"
#include "comm_utils_mpi.hpp"

namespace detail {

namespace gpump {

inline struct ::gpump* init(MPI_Comm mpi_comm)
{
  // LOGPRINTF("gpump_init rank(w%i)\n", MPI::Comm_rank(MPI_COMM_WORLD));
  struct ::gpump* g = gpump_init(mpi_comm);
  // LOGPRINTF("gpump_init rank(w%i) done -> %p\n", MPI::Comm_rank(MPI_COMM_WORLD), g);
  assert(g != nullptr);
  return g;
}

inline void term(struct ::gpump* g)
{
  // LOGPRINTF("gpump_term(%p) rank(w%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD));
  gpump_term(g);
}

inline void connect_propose(struct ::gpump* g, int target)
{
  // LOGPRINTF("gpump_connect_propose(%p) rank(w%i) %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), target);
  gpump_connect_propose(g, target);
}

inline void connect_accept(struct ::gpump* g, int target)
{
  // LOGPRINTF("gpump_connect_accept(%p) rank(w%i) %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), target);
  gpump_connect_accept(g, target);
}

inline void disconnect(struct ::gpump* g, int target)
{
  // LOGPRINTF("gpump_disconnect(%p) rank(w%i) %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), target);
  gpump_disconnect(g, target);
}

inline struct ::ibv_mr* register_region(struct ::gpump* g, void* ptr, size_t size)
{
  // LOGPRINTF("gpump_register_region(%p) rank(w%i) %p[%zu]\n", g, MPI::Comm_rank(MPI_COMM_WORLD), ptr, size);
  struct ::ibv_mr* mr = gpump_register_region(g, ptr, size);
  // LOGPRINTF("gpump_register_region(%p) rank(w%i) %p[%zu] done -> %p\n", g, MPI::Comm_rank(MPI_COMM_WORLD), ptr, size, mr);
  return mr;
}

inline void deregister_region(struct ::gpump* g, struct ::ibv_mr* mr)
{
  // LOGPRINTF("gpump_deregister_region(%p) rank(w%i) %p\n", g, MPI::Comm_rank(MPI_COMM_WORLD), mr);
  gpump_deregister_region(g, mr);
}

inline void cork(struct ::gpump* g)
{
  // LOGPRINTF("gpump_cork(%p) rank(w%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD));
  gpump_cork(g);
}

inline void uncork(struct ::gpump* g, cudaStream_t stream)
{
  // LOGPRINTF("gpump_uncork(%p) rank(w%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), (void*)stream);
  gpump_uncork(g, stream);
}

inline void receive(struct ::gpump* g, int src, struct ::ibv_mr* buf_mr, size_t offset, size_t size)
{
  // LOGPRINTF("gpump_receive(%p) rank(w%i) %p+%zu[%zu] src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), buf_mr, offset, size, src);
  gpump_receive(g, src, buf_mr, offset, size);
}

inline void stream_wait_recv_complete(struct ::gpump* g, int src, cudaStream_t stream)
{
  // LOGPRINTF("gpump_stream_wait_recv_complete(%p) rank(w%i) src(%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src, (void*)stream);
  gpump_stream_wait_recv_complete(g, src, stream);
}

inline void cpu_ack_recv(struct ::gpump* g, int src)
{
  // LOGPRINTF("gpump_cpu_ack_recv(%p) rank(w%i) src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src);
  gpump_cpu_ack_recv(g, src);
}

inline int is_receive_complete(struct ::gpump* g, int src)
{
  // LOGPRINTF("gpump_is_receive_complete(%p) rank(w%i) src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src);
  int complete = gpump_is_receive_complete(g, src);
  // LOGPRINTF("gpump_is_receive_complete(%p) rank(w%i) src(%i) done -> %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src, complete);
  return complete;
}

inline void wait_receive_complete(struct ::gpump* g, int src)
{
  // LOGPRINTF("gpump_wait_receive_complete(%p) rank(w%i) src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src);
  gpump_wait_receive_complete(g, src);
}

inline void stream_send(struct ::gpump* g, int dest, cudaStream_t stream, struct ::ibv_mr* buf_mr, size_t offset, size_t size)
{
  // LOGPRINTF("gpump_stream_send(%p) rank(w%i) %p+%zu[%zu] dst(%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), buf_mr, offset, size, dest, (void*)stream);
  gpump_stream_send(g, dest, stream, buf_mr, offset, size);
}

inline void isend(struct ::gpump* g, int dest, struct ::ibv_mr* buf_mr, size_t offset, size_t size)
{
  // LOGPRINTF("gpump_isend(%p) rank(w%i) %p+%zu[%zu] dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), buf_mr, offset, size, dest);
  gpump_isend(g, dest, buf_mr, offset, size);
}

inline void stream_wait_send_complete(struct ::gpump* g, int dest, cudaStream_t stream)
{
  // LOGPRINTF("gpump_stream_wait_send_complete(%p) rank(w%i) dst(%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest, stream);
  gpump_stream_wait_send_complete(g, dest, stream);
}

inline void cpu_ack_isend(struct ::gpump* g, int dest)
{
  // LOGPRINTF("gpump_cpu_ack_isend(%p) rank(w%i) dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest);
  gpump_cpu_ack_isend(g, dest);
}

inline int is_send_complete(struct ::gpump* g, int dest)
{
  // LOGPRINTF("gpump_is_send_complete(%p) rank(w%i) dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest);
  int complete = gpump_is_send_complete(g, dest);
  // LOGPRINTF("gpump_is_send_complete(%p) rank(w%i) dst(%i) done -> %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest, complete);
  return complete;
}

inline void wait_send_complete(struct ::gpump* g, int dest)
{
  // LOGPRINTF("gpump_wait_send_complete(%p) rank(w%i) dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest);
  gpump_wait_send_complete(g, dest);
}

} // namespace gpump

} // namespace detail

#endif // COMB_ENABLE_GPUMP

#endif // _UTILS_GPUMP_HPP

