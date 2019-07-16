//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

#ifndef _UTILS_MP_HPP
#define _UTILS_MP_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_MP

#include <libmp.h>

#include <cassert>
#include <cstdio>

#include "utils.hpp"
#include "utils_mpi.hpp"

namespace detail {

namespace mp {

inline struct ::mp* init(MPI_Comm mpi_comm)
{
  // FPRINTF(stdout, "mp_init rank(w%i)\n", MPI::Comm_rank(MPI_COMM_WORLD));
  struct ::mp* g = mp_init(mpi_comm);
  // FPRINTF(stdout, "mp_init rank(w%i) done -> %p\n", MPI::Comm_rank(MPI_COMM_WORLD), g);
  assert(g != nullptr);
  return g;
}

inline void term(struct ::mp* g)
{
  // FPRINTF(stdout, "mp_term(%p) rank(w%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD));
  mp_term(g);
}

inline void connect_propose(struct ::mp* g, int target)
{
  // FPRINTF(stdout, "mp_connect_propose(%p) rank(w%i) %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), target);
  mp_connect_propose(g, target);
}

inline void connect_accept(struct ::mp* g, int target)
{
  // FPRINTF(stdout, "mp_connect_accept(%p) rank(w%i) %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), target);
  mp_connect_accept(g, target);
}

inline void disconnect(struct ::mp* g, int target)
{
  // FPRINTF(stdout, "mp_disconnect(%p) rank(w%i) %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), target);
  mp_disconnect(g, target);
}

inline struct ::ibv_mr* register_region(struct ::mp* g, void* ptr, size_t size)
{
  // FPRINTF(stdout, "mp_register_region(%p) rank(w%i) %p[%zu]\n", g, MPI::Comm_rank(MPI_COMM_WORLD), ptr, size);
  struct ::ibv_mr* mr = mp_register_region(g, ptr, size);
  // FPRINTF(stdout, "mp_register_region(%p) rank(w%i) %p[%zu] done -> %p\n", g, MPI::Comm_rank(MPI_COMM_WORLD), ptr, size, mr);
  return mr;
}

inline void deregister_region(struct ::mp* g, struct ::ibv_mr* mr)
{
  // FPRINTF(stdout, "mp_deregister_region(%p) rank(w%i) %p\n", g, MPI::Comm_rank(MPI_COMM_WORLD), mr);
  mp_deregister_region(g, mr);
}

inline void cork(struct ::mp* g)
{
  // FPRINTF(stdout, "mp_cork(%p) rank(w%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD));
  mp_cork(g);
}

inline void uncork(struct ::mp* g, cudaStream_t stream)
{
  // FPRINTF(stdout, "mp_uncork(%p) rank(w%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), (void*)stream);
  mp_uncork(g, stream);
}

inline void receive(struct ::mp* g, int src, struct ::ibv_mr* buf_mr, size_t offset, size_t size)
{
  // FPRINTF(stdout, "mp_receive(%p) rank(w%i) %p+%zu[%zu] src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), buf_mr, offset, size, src);
  mp_receive(g, src, buf_mr, offset, size);
}

inline void stream_wait_recv_complete(struct ::mp* g, int src, cudaStream_t stream)
{
  // FPRINTF(stdout, "mp_stream_wait_recv_complete(%p) rank(w%i) src(%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src, (void*)stream);
  mp_stream_wait_recv_complete(g, src, stream);
}

inline void cpu_ack_recv(struct ::mp* g, int src)
{
  // FPRINTF(stdout, "mp_cpu_ack_recv(%p) rank(w%i) src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src);
  mp_cpu_ack_recv(g, src);
}

inline int is_receive_complete(struct ::mp* g, int src)
{
  // FPRINTF(stdout, "mp_is_receive_complete(%p) rank(w%i) src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src);
  int complete = mp_is_receive_complete(g, src);
  // FPRINTF(stdout, "mp_is_receive_complete(%p) rank(w%i) src(%i) done -> %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src, complete);
  return complete;
}

inline void wait_receive_complete(struct ::mp* g, int src)
{
  // FPRINTF(stdout, "mp_wait_receive_complete(%p) rank(w%i) src(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), src);
  mp_wait_receive_complete(g, src);
}

inline void stream_send(struct ::mp* g, int dest, cudaStream_t stream, struct ::ibv_mr* buf_mr, size_t offset, size_t size)
{
  // FPRINTF(stdout, "mp_stream_send(%p) rank(w%i) %p+%zu[%zu] dst(%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), buf_mr, offset, size, dest, (void*)stream);
  mp_stream_send(g, dest, stream, buf_mr, offset, size);
}

inline void isend(struct ::mp* g, int dest, struct ::ibv_mr* buf_mr, size_t offset, size_t size)
{
  // FPRINTF(stdout, "mp_isend(%p) rank(w%i) %p+%zu[%zu] dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), buf_mr, offset, size, dest);
  mp_isend(g, dest, buf_mr, offset, size);
}

inline void stream_wait_send_complete(struct ::mp* g, int dest, cudaStream_t stream)
{
  // FPRINTF(stdout, "mp_stream_wait_send_complete(%p) rank(w%i) dst(%i) stream(%p)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest, stream);
  mp_stream_wait_send_complete(g, dest, stream);
}

inline void cpu_ack_isend(struct ::mp* g, int dest)
{
  // FPRINTF(stdout, "mp_cpu_ack_isend(%p) rank(w%i) dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest);
  mp_cpu_ack_isend(g, dest);
}

inline int is_send_complete(struct ::mp* g, int dest)
{
  // FPRINTF(stdout, "mp_is_send_complete(%p) rank(w%i) dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest);
  int complete = mp_is_send_complete(g, dest);
  // FPRINTF(stdout, "mp_is_send_complete(%p) rank(w%i) dst(%i) done -> %i\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest, complete);
  return complete;
}

inline void wait_send_complete(struct ::mp* g, int dest)
{
  // FPRINTF(stdout, "mp_wait_send_complete(%p) rank(w%i) dst(%i)\n", g, MPI::Comm_rank(MPI_COMM_WORLD), dest);
  mp_wait_send_complete(g, dest);
}

} // namespace mp

} // namespace detail

#endif // COMB_ENABLE_MP

#endif // _UTILS_MP_HPP

