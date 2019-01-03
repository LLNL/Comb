//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

#ifndef _UTILS_HPP
#define _UTILS_HPP

#include "config.hpp"

#include <cassert>
#include <cstdio>

#include <mpi.h>

#include "cuda_utils.hpp"

#define COMB_SERIALIZE_HELPER(a) #a
#define COMB_SERIALIZE(a) COMB_SERIALIZE_HELPER(a)

using IdxT = int;
using LidxT = int;
using DataT = double;

#ifdef __CUDA_ARCH__
#define FFLUSH(f) static_cast<void>(0)
#else
#define FFLUSH(f) fflush(f)
#endif

#ifdef __CUDA_ARCH__
#define FPRINTF(f, ...) printf(__VA_ARGS__), FFLUSH(f)
#else
#define FPRINTF(f, ...) fprintf(f, __VA_ARGS__), FFLUSH(f)
#endif


namespace detail {

template < typename T, typename ... types >
struct Count;

template < typename T >
struct Count<T> {
  static const size_t value = 0;
};

template < typename T, typename ... types >
struct Count<T, T, types...> {
  static const size_t value = 1 + Count<T, types...>::value;
};

template < typename T, typename T0, typename ... types >
struct Count<T, T0, types...> {
  static const size_t value = Count<T, types...>::value;
};

namespace MPI {

inline int Comm_rank(MPI_Comm comm);

inline int Init_thread(int* argc, char***argv, int required)
{
  int provided = required;
  // FPRINTF(stdout, "MPI_Init_thread\n");
  int ret = MPI_Init_thread(argc, argv, required, &provided);
  // FPRINTF(stdout, "MPI_Init_thread done rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  assert(ret == MPI_SUCCESS);
  //assert(required == provided);
  return provided;
}

inline void Abort(MPI_Comm comm, int errorcode)
{
  // FPRINTF(stdout, "MPI_Abort\n");
  int ret = MPI_Abort(comm, errorcode);
  assert(ret == MPI_SUCCESS);
}

inline void Finalize()
{
  // FPRINTF(stdout, "MPI_Finalize\n");
  int ret = MPI_Finalize();
  assert(ret == MPI_SUCCESS);
}

inline MPI_Comm Comm_dup(MPI_Comm comm_old)
{
  MPI_Comm comm;
  // FPRINTF(stdout, "MPI_Comm_dup rank(w%i) %s\n", Comm_rank(MPI_COMM_WORLD), name);
  int ret = MPI_Comm_dup(comm_old, &comm);
  assert(ret == MPI_SUCCESS);
  return comm;
}

inline void Comm_set_name(MPI_Comm comm, const char* name)
{
  // FPRINTF(stdout, "MPI_Comm_set_name rank(w%i) %s\n", Comm_rank(MPI_COMM_WORLD), name);
  int ret = MPI_Comm_set_name(comm, name);
  assert(ret == MPI_SUCCESS);
}

inline int Comm_rank(MPI_Comm comm)
{
  int rank = -1;
  int ret = MPI_Comm_rank(comm, &rank);
  //int wrank = -1; int wret = MPI_Comm_rank(MPI_COMM_WORLD, &wrank); FPRINTF(stdout, "MPI_Comm_rank rank(w%i %i)\n", wrank, rank); assert(wret == MPI_SUCCESS);
  assert(ret == MPI_SUCCESS);
  return rank;
}

inline int Comm_size(MPI_Comm comm)
{
  int size = -1;
  int ret = MPI_Comm_size(comm, &size);
  // FPRINTF(stdout, "MPI_Comm_size rank(w%i) %i\n", Comm_rank(MPI_COMM_WORLD), size);
  assert(ret == MPI_SUCCESS);
  return size;
}

inline void Comm_free(MPI_Comm* comm)
{
  // FPRINTF(stdout, "MPI_Comm_free rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Comm_free(comm);
  assert(ret == MPI_SUCCESS);
}

inline MPI_Comm Cart_create(MPI_Comm comm_old, int ndims, const int*dims, const int* periods, int reorder)
{
  MPI_Comm cartcomm;
  // FPRINTF(stdout, "MPI_Cart_create rank(w%i) dims %i(%i %i %i) periods (%i %i %i) reorder %i\n", Comm_rank(MPI_COMM_WORLD), ndims, dims[0], dims[1], dims[2], periods[0], periods[1], periods[2], reorder);
  int ret = MPI_Cart_create(comm_old, ndims, dims, periods, reorder, &cartcomm);
  assert(ret == MPI_SUCCESS);
  return cartcomm;
}

inline void Cart_coords(MPI_Comm cartcomm, int rank, int maxdims, int* coords)
{

  int ret = MPI_Cart_coords(cartcomm, rank, maxdims, coords);
  // FPRINTF(stdout, "MPI_Cart_coords rank(w%i c%i) coords %i(%i %i %i)\n", Comm_rank(MPI_COMM_WORLD), rank, maxdims, coords[0], coords[1], coords[2]);
  assert(ret == MPI_SUCCESS);
}

inline int Cart_rank(MPI_Comm cartcomm, const int* coords)
{
  int rank = -1;
  int ret = MPI_Cart_rank(cartcomm, coords, &rank);
  // FPRINTF(stdout, "MPI_Cart_rank rank(w%i c%i) coords (%i %i %i)\n", Comm_rank(MPI_COMM_WORLD), rank, coords[0], coords[1], coords[2]);
  assert(ret == MPI_SUCCESS);
  return rank;
}

inline void Barrier(MPI_Comm comm)
{
  // FPRINTF(stdout, "MPI_Barrier rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Barrier(comm);
  assert(ret == MPI_SUCCESS);
}

inline void Irecv(void *buf, int nbytes, int src, int tag, MPI_Comm comm, MPI_Request *request)
{
  // FPRINTF(stdout, "MPI_Irecv rank(w%i) %p[%i] src(%i) tag(%i)\n", Comm_rank(MPI_COMM_WORLD), buf, nbytes, src, tag);
  int ret = MPI_Irecv(buf, nbytes, MPI_BYTE, src, tag, comm, request);
  assert(ret == MPI_SUCCESS);
}

inline void Isend(const void *buf, int nbytes, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  // FPRINTF(stdout, "MPI_Isend rank(w%i) %p[%i] dst(%i) tag(%i)\n", Comm_rank(MPI_COMM_WORLD), buf, nbytes, dest, tag);
  int ret = MPI_Isend(buf, nbytes, MPI_BYTE, dest, tag, comm, request);
  assert(ret == MPI_SUCCESS);
}

inline void Wait(MPI_Request *request, MPI_Status *status)
{
  // FPRINTF(stdout, "MPI_Wait rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Wait(request, status);
  assert(ret == MPI_SUCCESS);
}

inline bool Test(MPI_Request *request, MPI_Status *status)
{
  int completed = 0;
  // FPRINTF(stdout, "MPI_Test rank(w%i)\n", Comm_rank(MPI_COMM_WORLD));
  int ret = MPI_Test(request, &completed, status);
  assert(ret == MPI_SUCCESS);
  return completed;
}

inline int Waitany(int count, MPI_Request *requests, MPI_Status *status)
{
  int idx = -1;
  // FPRINTF(stdout, "MPI_Waitany rank(w%i) count(%i)\n", Comm_rank(MPI_COMM_WORLD), count);
  int ret = MPI_Waitany(count, requests, &idx, status);
  assert(ret == MPI_SUCCESS);
  return idx;
}

inline int Testany(int count, MPI_Request *requests, MPI_Status *status)
{
  int completed = 0;
  int indx = -1;
  // FPRINTF(stdout, "MPI_Testany rank(w%i) count(%i)\n", Comm_rank(MPI_COMM_WORLD), count);
  int ret = MPI_Testany(count, requests, &indx, &completed, status);
  assert(ret == MPI_SUCCESS);
  return completed ? indx : -1;
}

inline int Waitsome(int incount, MPI_Request *requests, int* indcs, MPI_Status *statuses)
{
  int outcount = 0;
  // FPRINTF(stdout, "MPI_Waitsome rank(w%i) incount(%i)\n", Comm_rank(MPI_COMM_WORLD), incount);
  int ret = MPI_Waitsome(incount, requests, &outcount, indcs, statuses);
  assert(ret == MPI_SUCCESS);
  return outcount;
}

inline int Testsome(int incount, MPI_Request *requests, int* indcs, MPI_Status *statuses)
{
  int outcount = 0;
  // FPRINTF(stdout, "MPI_Testsome rank(w%i) incount(%i)\n", Comm_rank(MPI_COMM_WORLD), incount);
  int ret = MPI_Testsome(incount, requests, &outcount, indcs, statuses);
  assert(ret == MPI_SUCCESS);
  return outcount;
}

inline void Waitall(int count, MPI_Request *requests, MPI_Status *statuses)
{
  // FPRINTF(stdout, "MPI_Waitall rank(w%i) count(%i)\n", Comm_rank(MPI_COMM_WORLD), count);
  int ret = MPI_Waitall(count, requests, statuses);
  assert(ret == MPI_SUCCESS);
}

inline bool Testall(int count, MPI_Request *requests, MPI_Status *statuses)
{
  int completed = 0;
  // FPRINTF(stdout, "MPI_Testall rank(w%i) count(%i)\n", Comm_rank(MPI_COMM_WORLD), count);
  int ret = MPI_Testall(count, requests, &completed, statuses);
  assert(ret == MPI_SUCCESS);
  return completed;
}

} // namespace MPI


struct indexer_kji {
  IdxT ijlen, ilen;
  indexer_kji(IdxT ijlen_, IdxT ilen_) : ijlen(ijlen_), ilen(ilen_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT j, IdxT i, IdxT) const { return i + j * ilen + k * ijlen; }
};
struct indexer_ji {
  IdxT ilen, koff;
  indexer_ji(IdxT ilen_, IdxT koff_) : ilen(ilen_), koff(koff_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT j, IdxT i, IdxT) const { return i + j * ilen + koff; }
};
struct indexer_ki {
  IdxT ijlen, joff;
  indexer_ki(IdxT ijlen_, IdxT joff_) : ijlen(ijlen_), joff(joff_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT i, IdxT) const { return i + joff + k * ijlen; }
};
struct indexer_kj {
  IdxT ijlen, ilen, ioff;
  indexer_kj(IdxT ijlen_, IdxT ilen_, IdxT ioff_) : ijlen(ijlen_), ilen(ilen_), ioff(ioff_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT j, IdxT) const { return ioff + j * ilen + k * ijlen; }
};

struct indexer_i {
  IdxT off;
  indexer_i(IdxT off_) : off(off_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT i, IdxT) const { return i + off; }
};
struct indexer_j {
  IdxT ilen, off;
  indexer_j(IdxT ilen_, IdxT off_) : ilen(ilen_), off(off_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT j, IdxT) const { return j * ilen + off; }
};
struct indexer_k {
  IdxT ijlen, off;
  indexer_k(IdxT ijlen_, IdxT off_) : ijlen(ijlen_), off(off_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT k, IdxT) const { return k * ijlen + off; }
};

struct indexer_ {
  IdxT off;
  indexer_(IdxT off_) : off(off_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT) const { return off; }
};

struct indexer_idx {
  indexer_idx() {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT idx) const { return idx; }
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT, IdxT idx) const { return idx; }
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT, IdxT, IdxT idx) const { return idx; }
};

struct indexer_list_idx {
  LidxT const* indices;
  indexer_list_idx(LidxT const* indices_) : indices(indices_) {}
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT idx) const { return indices[idx]; }
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT, IdxT idx) const { return indices[idx]; }
  COMB_HOST COMB_DEVICE IdxT operator()(IdxT, IdxT, IdxT, IdxT idx) const { return indices[idx]; }
};

template < typename T_src, typename I_src, typename T_dst, typename I_dst >
struct copy_idxr_idxr {
  T_src const* ptr_src;
  T_dst* ptr_dst;
  I_src idxr_src;
  I_dst idxr_dst;
  copy_idxr_idxr(T_src const* const& ptr_src_, I_src const& idxr_src_, T_dst* const& ptr_dst_, I_dst const& idxr_dst_) : ptr_src(ptr_src_), ptr_dst(ptr_dst_), idxr_src(idxr_src_), idxr_dst(idxr_dst_) {}
  template < typename ... Ts >
  COMB_HOST COMB_DEVICE void operator()(Ts... args) const
  {
    IdxT dst_i = idxr_dst(args...);
    IdxT src_i = idxr_src(args...);
    // FPRINTF(stdout, "copy_idxr_idxr %p[%i]{%f} = %p[%i]{%f} (%i)%i\n", ptr_dst, dst_i, (double)ptr_dst[dst_i],
    //                                                                    ptr_src, src_i, (double)ptr_src[src_i], args...);
    ptr_dst[dst_i] = ptr_src[src_i];
  }
};

template < typename T_src, typename I_src, typename T_dst, typename I_dst >
copy_idxr_idxr<T_src, I_src, T_dst, I_dst> make_copy_idxr_idxr(T_src* const& ptr_src, I_src const& idxr_src, T_dst* const& ptr_dst, I_dst const& idxr_dst) {
  return copy_idxr_idxr<T_src, I_src, T_dst, I_dst>(ptr_src, idxr_src, ptr_dst, idxr_dst);
}

template < typename I_src, typename T_dst, typename I_dst >
struct set_idxr_idxr {
  T_dst* ptr_dst;
  I_src idxr_src;
  I_dst idxr_dst;
  set_idxr_idxr(I_src const& idxr_src_, T_dst* const& ptr_dst_, I_dst const& idxr_dst_)
    : ptr_dst(ptr_dst_)
    , idxr_src(idxr_src_)
    , idxr_dst(idxr_dst_)
  { }
  template < typename ... Ts >
  COMB_HOST COMB_DEVICE void operator()(Ts... args) const
  {
    IdxT dst_i = idxr_dst(args...);
    IdxT src_i = idxr_src(args...);
    // FPRINTF(stdout, "set_idxr_idxr %p[%i]{%f} = %i (%i %i %i)%i\n", ptr_dst, dst_i, (double)ptr_dst[dst_i], src_i, args...);
    ptr_dst[dst_i] = src_i;
  }
};

template < typename I_src, typename T_dst, typename I_dst >
set_idxr_idxr<I_src, T_dst, I_dst> make_set_idxr_idxr(I_src const& idxr_src, T_dst* const& ptr_dst, I_dst const& idxr_dst) {
  return set_idxr_idxr<I_src, T_dst, I_dst>(idxr_src, ptr_dst, idxr_dst);
}

} // namespace detail

#endif // _UTILS_HPP

