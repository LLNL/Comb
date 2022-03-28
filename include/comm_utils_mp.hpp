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

#ifndef _UTILS_MP_HPP
#define _UTILS_MP_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_MP

#include <mp.h>

#include <cassert>
#include <cstdio>
#include <vector>
#include <numeric>

#include "exec_utils.hpp"
#include "exec_utils_cuda.hpp"
#include "comm_utils_mpi.hpp"

namespace detail {

namespace mp {

inline void init(MPI_Comm mpi_comm)
{
  // LOGPRINTF("mp_init rank(w%i)\n", MPI::Comm_rank(MPI_COMM_WORLD));
  int nranks = MPI::Comm_size(mpi_comm);
  std::vector<int> ranks(nranks);
  std::iota(ranks.begin(), ranks.end(), 0);
  int gpuid = COMB::detail::cuda::get_device();
  auto ret = mp_init(mpi_comm, ranks.data(), nranks, MP_INIT_DEFAULT, gpuid);
  // LOGPRINTF("mp_init rank(w%i) done\n", MPI::Comm_rank(MPI_COMM_WORLD));
  assert(ret == MP_SUCCESS);
}

inline void finalize()
{
  // LOGPRINTF("mp_finalize() rank(w%i)\n", MPI::Comm_rank(MPI_COMM_WORLD));
  mp_finalize();
}

inline mp_reg_t register_(void* ptr, size_t size)
{
  mp_reg_t reg;
  // LOGPRINTF("mp_register() rank(w%i) %p[%zu]\n", MPI::Comm_rank(MPI_COMM_WORLD), ptr, size);
  auto ret = mp_register(ptr, size, &reg);
  // LOGPRINTF("mp_register() rank(w%i) %p[%zu] done -> %p\n", MPI::Comm_rank(MPI_COMM_WORLD), ptr, size, (void*)reg);
  assert(ret == MP_SUCCESS);
  return reg;
}

inline void deregister(mp_reg_t& reg)
{
  // LOGPRINTF("mp_deregister() rank(w%i) %p\n", MPI::Comm_rank(MPI_COMM_WORLD), (void*)reg);
  auto ret = mp_deregister(&reg);
  assert(ret == MP_SUCCESS);
  reg = nullptr;
}

inline void irecv(void *buf, size_t size, int src, mp_reg_t* reg, mp_request_t *req)
{
  // LOGPRINTF("mp_irecv() rank(w%i) %p[%zu] src(%i) reg(%p) req(%p)\n", MPI::Comm_rank(MPI_COMM_WORLD), buf, size, src, reg, req);
  auto ret = mp_irecv(buf, size, src, reg, req);
  assert(ret == MP_SUCCESS);
}

inline void isend(void *buf, size_t size, int src, mp_reg_t* reg, mp_request_t *req)
{
  // LOGPRINTF("mp_isend() rank(w%i) %p[%zu] src(%i) reg(%p) req(%p)\n", MPI::Comm_rank(MPI_COMM_WORLD), buf, size, src, reg, req);
  auto ret = mp_isend(buf, size, src, reg, req);
  assert(ret == MP_SUCCESS);
}

inline void send_on_stream(void *buf, size_t size, int dst, mp_reg_t* reg, mp_request_t *req, cudaStream_t stream)
{
  // LOGPRINTF("mp_send_on_stream() rank(w%i) %p[%zu] dst(%i) reg(%p) req(%p) stream(%p)\n", MPI::Comm_rank(MPI_COMM_WORLD), buf, size, dst, reg, req, (void*)stream);
  auto ret = mp_send_on_stream(buf, size, dst, reg, req, stream);
  assert(ret == MP_SUCCESS);
}

inline void isend_on_stream(void *buf, size_t size, int dst, mp_reg_t* reg, mp_request_t *req, cudaStream_t stream)
{
  // LOGPRINTF("mp_isend_on_stream() rank(w%i) %p[%zu] dst(%i) reg(%p) req(%p) stream(%p)\n", MPI::Comm_rank(MPI_COMM_WORLD), buf, size, dst, reg, req, (void*)stream);
  auto ret = mp_isend_on_stream(buf, size, dst, reg, req, stream);
  assert(ret == MP_SUCCESS);
}

inline void wait_on_stream(mp_request_t *req, cudaStream_t stream)
{
  // LOGPRINTF("mp_wait_on_stream() rank(w%i) req(%p) stream(%p)\n", MPI::Comm_rank(MPI_COMM_WORLD), req, (void*)stream);
  auto ret = mp_wait_on_stream(req, stream);
  assert(ret == MP_SUCCESS);
}

inline void wait_all_on_stream(size_t count, mp_request_t *req, cudaStream_t stream)
{
  // LOGPRINTF("mp_wait_all_on_stream() rank(w%i) count (%zu) req(%p) stream(%p)\n", MPI::Comm_rank(MPI_COMM_WORLD), count, req, (void*)stream);
  auto ret = mp_wait_all_on_stream(count, req, stream);
  assert(ret == MP_SUCCESS);
}

inline void wait(mp_request_t *req)
{
  // LOGPRINTF("mp_wait() rank(w%i) req(%p)\n", MPI::Comm_rank(MPI_COMM_WORLD), req);
  auto ret = mp_wait(req);
  assert(ret == MP_SUCCESS);
}

inline void wait_all(size_t count, mp_request_t *req)
{
  // LOGPRINTF("mp_wait_all() rank(w%i) count (%zu) req(%p)\n", MPI::Comm_rank(MPI_COMM_WORLD), count, req);
  auto ret = mp_wait_all(count, req);
  assert(ret == MP_SUCCESS);
}

inline void progress_all(size_t count, mp_request_t *req)
{
  // LOGPRINTF("mp_progress_all() rank(w%i) count (%zu) req(%p)\n", MPI::Comm_rank(MPI_COMM_WORLD), count, req);
  auto ret = mp_progress_all(count, req);
  assert(ret == MP_SUCCESS);
}

} // namespace mp

} // namespace detail

#endif // COMB_ENABLE_MP

#endif // _UTILS_MP_HPP

