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

#ifndef _UTILS_MPI_HPP
#define _UTILS_MPI_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_MPI

#include <cassert>
#include <cstdio>

#include <mpi.h>

namespace detail {

namespace MPI {

inline int Comm_rank(MPI_Comm comm);

inline int Init_thread(int* argc, char***argv, int required)
{
  int provided = required;
  // LOGPRINTF("MPI_Init_thread\n");
  int ret = MPI_Init_thread(argc, argv, required, &provided);
  // LOGPRINTF("MPI_Init_thread done rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  assert(ret == MPI_SUCCESS);
  //assert(required == provided);
  return provided;
}

inline void Abort(MPI_Comm comm, int errorcode)
{
  // LOGPRINTF("MPI_Abort\n");
  int ret = MPI_Abort(comm, errorcode);
  assert(ret == MPI_SUCCESS);
}

inline void Finalize()
{
  // LOGPRINTF("MPI_Finalize\n");
  int ret = MPI_Finalize();
  assert(ret == MPI_SUCCESS);
}

inline MPI_Comm Comm_dup(MPI_Comm comm_old)
{
  MPI_Comm comm;
  // LOGPRINTF("MPI_Comm_dup rank(w%i) %s\n", Comm_rank(MPI_COMM_WORLD), name);
  int ret = MPI_Comm_dup(comm_old, &comm);
  assert(ret == MPI_SUCCESS);
  return comm;
}

inline void Comm_set_name(MPI_Comm comm, const char* name)
{
  // LOGPRINTF("MPI_Comm_set_name rank(w%i) %s\n", Comm_rank(MPI_COMM_WORLD), name);
  int ret = MPI_Comm_set_name(comm, name);
  assert(ret == MPI_SUCCESS);
}

inline int Comm_rank(MPI_Comm comm)
{
  int rank = -1;
  int ret = MPI_Comm_rank(comm, &rank);
  //int wrank = -1; int wret = MPI_Comm_rank(MPI_COMM_WORLD, &wrank); LOGPRINTF("MPI_Comm_rank rank(w%i %i)\n", wrank, rank); assert(wret == MPI_SUCCESS);
  assert(ret == MPI_SUCCESS);
  return rank;
}

inline int Comm_size(MPI_Comm comm)
{
  int size = -1;
  int ret = MPI_Comm_size(comm, &size);
  // LOGPRINTF("MPI_Comm_size rank(w%i) %i\n", Comm_rank(MPI_COMM_WORLD), size);
  assert(ret == MPI_SUCCESS);
  return size;
}

inline void Comm_free(MPI_Comm* comm)
{
  // LOGPRINTF("MPI_Comm_free rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Comm_free(comm);
  assert(ret == MPI_SUCCESS);
}

inline MPI_Comm Cart_create(MPI_Comm comm_old, int ndims, const int*dims, const int* periods, int reorder)
{
  MPI_Comm cartcomm;
  // LOGPRINTF("MPI_Cart_create rank(w%i) dims %i(%i %i %i) periods (%i %i %i) reorder %i\n", Comm_rank(MPI_COMM_WORLD), ndims, dims[0], dims[1], dims[2], periods[0], periods[1], periods[2], reorder);
  int ret = MPI_Cart_create(comm_old, ndims, dims, periods, reorder, &cartcomm);
  assert(ret == MPI_SUCCESS);
  return cartcomm;
}

inline void Cart_coords(MPI_Comm cartcomm, int rank, int maxdims, int* coords)
{

  int ret = MPI_Cart_coords(cartcomm, rank, maxdims, coords);
  // LOGPRINTF("MPI_Cart_coords rank(w%i c%i) coords %i(%i %i %i)\n", Comm_rank(MPI_COMM_WORLD), rank, maxdims, coords[0], coords[1], coords[2]);
  assert(ret == MPI_SUCCESS);
}

inline int Cart_rank(MPI_Comm cartcomm, const int* coords)
{
  int rank = -1;
  int ret = MPI_Cart_rank(cartcomm, coords, &rank);
  // LOGPRINTF("MPI_Cart_rank rank(w%i c%i) coords (%i %i %i)\n", Comm_rank(MPI_COMM_WORLD), rank, coords[0], coords[1], coords[2]);
  assert(ret == MPI_SUCCESS);
  return rank;
}

inline MPI_Datatype Type_create_indexed_block(int count, int blocklength, const int *displacements, MPI_Datatype old_type)
{
  MPI_Datatype mpi_type;
  int ret = MPI_Type_create_indexed_block(count, blocklength, displacements, old_type, &mpi_type);
  // LOGPRINTF("MPI_Type_create_indexed_block rank(w%i) count(%i) blocklength(%i) displacements(%p)\n", Comm_rank(MPI_COMM_WORLD), count, blocklength, displacements);
  assert(ret == MPI_SUCCESS);
  return mpi_type;
}

inline MPI_Datatype Type_create_subarray(int ndims, const int *sizes, const int *subsizes, const int *starts, int order, MPI_Datatype old_type)
{
  MPI_Datatype mpi_type;
  int ret = MPI_Type_create_subarray(ndims, sizes, subsizes, starts, order, old_type, &mpi_type);
  // LOGPRINTF("MPI_Type_create_subarray rank(w%i) ndims(%i) sizes(%i %i %i) subsizes(%i %i %i) starts(%i %i %i) order(%i)\n", Comm_rank(MPI_COMM_WORLD), ndims, sizes[0], sizes[1], sizes[2], subsizes[0], subsizes[1], subsizes[2], starts[0], starts[1], starts[2], order);
  assert(ret == MPI_SUCCESS);
  return mpi_type;
}

inline void Type_commit(MPI_Datatype* mpi_type)
{
  int ret = MPI_Type_commit(mpi_type);
  // LOGPRINTF("MPI_Type_commit rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  assert(ret == MPI_SUCCESS);
}

inline void Type_free(MPI_Datatype* mpi_type)
{
  int ret = MPI_Type_free(mpi_type);
  // LOGPRINTF("MPI_Type_free rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  assert(ret == MPI_SUCCESS);
}

inline void Barrier(MPI_Comm comm)
{
  // LOGPRINTF("MPI_Barrier rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Barrier(comm);
  assert(ret == MPI_SUCCESS);
}

inline void Bcast(void* buf, int count, MPI_Datatype mpi_type, int root, MPI_Comm comm)
{
  // LOGPRINTF("MPI_Bcast rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Bcast(buf, count, mpi_type, root, comm);
  assert(ret == MPI_SUCCESS);
}

inline void Reduce(const void* inbuf, void* outbuf, int count, MPI_Datatype mpi_type, MPI_Op op, int root, MPI_Comm comm)
{
  // LOGPRINTF("MPI_Reduce rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Reduce(inbuf, outbuf, count, mpi_type, op, root, comm);
  assert(ret == MPI_SUCCESS);
}

inline int Pack_size(int incount, MPI_Datatype mpi_type, MPI_Comm comm)
{
  int size;
  int ret = MPI_Pack_size(incount, mpi_type, comm, &size);
  // LOGPRINTF("MPI_Pack_size rank(w%i) incount(%i) size (%i)\n", Comm_rank(MPI_COMM_WORLD), buf, incount, size);
  assert(ret == MPI_SUCCESS);
  return size;
}

inline void Pack(const void* inbuf, int incount, MPI_Datatype mpi_type, void* outbuf, int outsize, int* position, MPI_Comm comm)
{
  // int inposition = *position;
  int ret = MPI_Pack(inbuf, incount, mpi_type, outbuf, outsize, position, comm);
  // LOGPRINTF("MPI_Pack rank(w%i) %p[%i] = %p[%i] position(%i -> %i)\n", Comm_rank(MPI_COMM_WORLD), outbuf, outsize, inbuf, incount, inposition, *position);
  assert(ret == MPI_SUCCESS);
}

inline void Unpack(const void* inbuf, int insize, int* position, void* outbuf, int outcount, MPI_Datatype mpi_type, MPI_Comm comm)
{
  // int inposition = *position;
  int ret = MPI_Unpack(inbuf, insize, position, outbuf, outcount, mpi_type, comm);
  // LOGPRINTF("MPI_Unpack rank(w%i) %p[%i] = %p[%i] position(%i -> %i)\n", Comm_rank(MPI_COMM_WORLD), outbuf, outcount, inbuf, insize, inposition, *position);
  assert(ret == MPI_SUCCESS);
}

inline void Irecv(void *buf, int count, MPI_Datatype mpi_type, int src, int tag, MPI_Comm comm, MPI_Request *request)
{
  // LOGPRINTF("MPI_Irecv rank(w%i) %p[%i] src(%i) tag(%i)\n", Comm_rank(MPI_COMM_WORLD), buf, count, src, tag);
  int ret = MPI_Irecv(buf, count, mpi_type, src, tag, comm, request);
  assert(ret == MPI_SUCCESS);
}

inline void Isend(const void *buf, int count, MPI_Datatype mpi_type, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  // LOGPRINTF("MPI_Isend rank(w%i) %p[%i] dst(%i) tag(%i)\n", Comm_rank(MPI_COMM_WORLD), buf, count, dest, tag);
  int ret = MPI_Isend(buf, count, mpi_type, dest, tag, comm, request);
  assert(ret == MPI_SUCCESS);
}

inline void Recv_init(void *buf, int count, MPI_Datatype mpi_type, int src, int tag, MPI_Comm comm, MPI_Request *request)
{
  int ret = MPI_Recv_init(buf, count, mpi_type, src, tag, comm, request);
  assert(ret == MPI_SUCCESS);
}

inline void Send_init(void *buf, int count, MPI_Datatype mpi_type, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  int ret = MPI_Send_init(buf, count, mpi_type, dest, tag, comm, request);
  assert(ret == MPI_SUCCESS);
}

inline void Start(MPI_Request *request)
{
  int ret = MPI_Start(request);
  assert(ret == MPI_SUCCESS);
}

inline void Startall(int count, MPI_Request *requests)
{
  int ret = MPI_Startall(count, requests);
  assert(ret == MPI_SUCCESS);
}

inline void Wait(MPI_Request *request, MPI_Status *status)
{
  // LOGPRINTF("MPI_Wait rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Wait(request, status);
  assert(ret == MPI_SUCCESS);
}

inline bool Test(MPI_Request *request, MPI_Status *status)
{
  int completed = 0;
  // LOGPRINTF("MPI_Test rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Test(request, &completed, status);
  assert(ret == MPI_SUCCESS);
  return completed;
}

inline int Waitany(int count, MPI_Request *requests, MPI_Status *status)
{
  int idx = -1;
  // LOGPRINTF("MPI_Waitany rank(w%i) count(%i)\n", Comm_rank(MPI_COMM_WORLD), count);
  int ret = MPI_Waitany(count, requests, &idx, status);
  assert(ret == MPI_SUCCESS);
  return idx;
}

inline int Testany(int count, MPI_Request *requests, MPI_Status *status)
{
  int completed = 0;
  int indx = -1;
  // LOGPRINTF("MPI_Testany rank(w%i) count(%i)\n", Comm_rank(MPI_COMM_WORLD), count);
  int ret = MPI_Testany(count, requests, &indx, &completed, status);
  assert(ret == MPI_SUCCESS);
  return completed ? indx : -1;
}

inline int Waitsome(int incount, MPI_Request *requests, int* indcs, MPI_Status *statuses)
{
  int outcount = 0;
  // LOGPRINTF("MPI_Waitsome rank(w%i) incount(%i)\n", Comm_rank(MPI_COMM_WORLD), incount);
  int ret = MPI_Waitsome(incount, requests, &outcount, indcs, statuses);
  assert(ret == MPI_SUCCESS);
  return outcount;
}

inline int Testsome(int incount, MPI_Request *requests, int* indcs, MPI_Status *statuses)
{
  int outcount = 0;
  // LOGPRINTF("MPI_Testsome rank(w%i) incount(%i)\n", Comm_rank(MPI_COMM_WORLD), incount);
  int ret = MPI_Testsome(incount, requests, &outcount, indcs, statuses);
  assert(ret == MPI_SUCCESS);
  return outcount;
}

inline void Waitall(int count, MPI_Request *requests, MPI_Status *statuses)
{
  // LOGPRINTF("MPI_Waitall rank(w%i) count(%i)\n", Comm_rank(MPI_COMM_WORLD), count);
  int ret = MPI_Waitall(count, requests, statuses);
  assert(ret == MPI_SUCCESS);
}

inline bool Testall(int count, MPI_Request *requests, MPI_Status *statuses)
{
  int completed = 0;
  // LOGPRINTF("MPI_Testall rank(w%i) count(%i)\n", Comm_rank(MPI_COMM_WORLD), count);
  int ret = MPI_Testall(count, requests, &completed, statuses);
  assert(ret == MPI_SUCCESS);
  return completed;
}

inline void Request_free(MPI_Request *request)
{
  int ret = MPI_Request_free(request);
  assert(ret == MPI_SUCCESS);
}

} // namespace MPI

} // namespace detail

#endif

#endif // _UTILS_MPI_HPP

