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

#ifndef _UTILS_GDSYNC_HPP
#define _UTILS_GDSYNC_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_GDSYNC

#include <libgdsync.h>

#include <cassert>
#include <cstdio>

#include "utils.hpp"
#include "utils_mpi.hpp"

namespace detail {

namespace gdsync {

inline struct ::gdsync* init(MPI_Comm mpi_comm)
{
  // FGPRINTF(FileGroup::proc, "gdsync_init rank(w%i)\n", MPI::Comm_rank(MPI_COMM_WORLD));
  struct ::gdsync* g = gdsync_init(mpi_comm);
  // FGPRINTF(FileGroup::proc, "gdsync_init rank(w%i) done -> %p\n", MPI::Comm_rank(MPI_COMM_WORLD), g);
  assert(g != nullptr);
  return g;
}

inline void term(struct ::gdsync* g)
{
  // FGPRINTF(FileGroup::proc, "gdsync_term(%p) rank(w%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD));
  gdsync_term(g);
}

inline void connect_propose(struct ::gdsync* g, int target)
{
  // FGPRINTF(FileGroup::proc, "gdsync_connect_propose(%p) rank(w%i) %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), target);
  gdsync_connect_propose(g, target);
}

inline void connect_accept(struct ::gdsync* g, int target)
{
  // FGPRINTF(FileGroup::proc, "gdsync_connect_accept(%p) rank(w%i) %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), target);
  gdsync_connect_accept(g, target);
}

inline void disconnect(struct ::gdsync* g, int target)
{
  // FGPRINTF(FileGroup::proc, "gdsync_disconnect(%p) rank(w%i) %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), target);
  gdsync_disconnect(g, target);
}

inline struct ::ibv_mr* register_region(struct ::gdsync* g, void* ptr, size_t size)
{
  // FGPRINTF(FileGroup::proc, "gdsync_register_region(%p) rank(w%i) %p[%zu]\n", g, MPI::Comm_rank(MPI_COMM_WORLD), ptr, size);
  struct ::ibv_mr* mr = gdsync_register_region(g, ptr, size);
  // FGPRINTF(FileGroup::proc, "gdsync_register_region(%p) rank(w%i) %p[%zu] done -> %p\n", g, MPI::Comm_rank(MPI_COMM_WORLD), ptr, size, mr);
  return mr;
}

inline void deregister_region(struct ::gdsync* g, struct ::ibv_mr* mr)
{
  // FGPRINTF(FileGroup::proc, "gdsync_deregister_region(%p) rank(w%i) %p\n", g, MPI::Comm_rank(MPI_COMM_WORLD), mr);
  gdsync_deregister_region(g, mr);
}

inline void cork(struct ::gdsync* g)
{
  // FGPRINTF(FileGroup::proc, "gdsync_cork(%p) rank(w%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD));
  gdsync_cork(g);
}

inline void uncork(struct ::gdsync* g, cudaStream_t stream)
{
  // FGPRINTF(FileGroup::proc, "gdsync_uncork(%p) rank(w%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), (void*)stream);
  gdsync_uncork(g, stream);
}

inline void receive(struct ::gdsync* g, int src, struct ::ibv_mr* buf_mr, size_t offset, size_t size)
{
  // FGPRINTF(FileGroup::proc, "gdsync_receive(%p) rank(w%i) %p+%zu[%zu] src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), buf_mr, offset, size, src);
  gdsync_receive(g, src, buf_mr, offset, size);
}

inline void stream_wait_recv_complete(struct ::gdsync* g, int src, cudaStream_t stream)
{
  // FGPRINTF(FileGroup::proc, "gdsync_stream_wait_recv_complete(%p) rank(w%i) src(%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src, (void*)stream);
  gdsync_stream_wait_recv_complete(g, src, stream);
}

inline void cpu_ack_recv(struct ::gdsync* g, int src)
{
  // FGPRINTF(FileGroup::proc, "gdsync_cpu_ack_recv(%p) rank(w%i) src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src);
  gdsync_cpu_ack_recv(g, src);
}

inline int is_receive_complete(struct ::gdsync* g, int src)
{
  // FGPRINTF(FileGroup::proc, "gdsync_is_receive_complete(%p) rank(w%i) src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src);
  int complete = gdsync_is_receive_complete(g, src);
  // FGPRINTF(FileGroup::proc, "gdsync_is_receive_complete(%p) rank(w%i) src(%i) done -> %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src, complete);
  return complete;
}

inline void wait_receive_complete(struct ::gdsync* g, int src)
{
  // FGPRINTF(FileGroup::proc, "gdsync_wait_receive_complete(%p) rank(w%i) src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src);
  gdsync_wait_receive_complete(g, src);
}

inline void stream_send(struct ::gdsync* g, int dest, cudaStream_t stream, struct ::ibv_mr* buf_mr, size_t offset, size_t size)
{
  // FGPRINTF(FileGroup::proc, "gdsync_stream_send(%p) rank(w%i) %p+%zu[%zu] dst(%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), buf_mr, offset, size, dest, (void*)stream);
  gdsync_stream_send(g, dest, stream, buf_mr, offset, size);
}

inline void isend(struct ::gdsync* g, int dest, struct ::ibv_mr* buf_mr, size_t offset, size_t size)
{
  // FGPRINTF(FileGroup::proc, "gdsync_isend(%p) rank(w%i) %p+%zu[%zu] dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), buf_mr, offset, size, dest);
  gdsync_isend(g, dest, buf_mr, offset, size);
}

inline void stream_wait_send_complete(struct ::gdsync* g, int dest, cudaStream_t stream)
{
  // FGPRINTF(FileGroup::proc, "gdsync_stream_wait_send_complete(%p) rank(w%i) dst(%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest, stream);
  gdsync_stream_wait_send_complete(g, dest, stream);
}

inline void cpu_ack_isend(struct ::gdsync* g, int dest)
{
  // FGPRINTF(FileGroup::proc, "gdsync_cpu_ack_isend(%p) rank(w%i) dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest);
  gdsync_cpu_ack_isend(g, dest);
}

inline int is_send_complete(struct ::gdsync* g, int dest)
{
  // FGPRINTF(FileGroup::proc, "gdsync_is_send_complete(%p) rank(w%i) dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest);
  int complete = gdsync_is_send_complete(g, dest);
  // FGPRINTF(FileGroup::proc, "gdsync_is_send_complete(%p) rank(w%i) dst(%i) done -> %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest, complete);
  return complete;
}

inline void wait_send_complete(struct ::gdsync* g, int dest)
{
  // FGPRINTF(FileGroup::proc, "gdsync_wait_send_complete(%p) rank(w%i) dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest);
  gdsync_wait_send_complete(g, dest);
}

} // namespace gdsync

} // namespace detail

#endif // COMB_ENABLE_GDSYNC

#endif // _UTILS_GDSYNC_HPP

