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

#ifndef _UTILS_UMR_HPP
#define _UTILS_UMR_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_UMR

#include <cassert>
#include <cstdio>

#include <umr.h>

namespace detail {

namespace UMR {

inline int Init_thread(int* argc, char***argv, int required)
{
  int provided = required;
  // FPRINTF(stdout, "UMR_Init_thread\n");
  int ret = UMR_Init_thread(argc, argv, required, &provided);
  // FPRINTF(stdout, "UMR_Init_thread done rank(w%i)\n", Comm_rank(UMR_COMM_WORLD));
  assert(ret == UMR_SUCCESS);
  //assert(required == provided);
  return provided;
}

inline void Finalize()
{
  // FPRINTF(stdout, "UMR_Finalize\n");
  int ret = UMR_Finalize();
  assert(ret == UMR_SUCCESS);
}

inline void Irecv(void *buf, int count, UMR_Datatype umr_type, int src, int tag, UMR_Comm comm, UMR_Request *request)
{
  // FPRINTF(stdout, "UMR_Irecv rank(w%i) %p[%i] src(%i) tag(%i)\n", Comm_rank(UMR_COMM_WORLD), buf, count, src, tag);
  int ret = UMR_Irecv(buf, count, umr_type, src, tag, comm, request);
  assert(ret == UMR_SUCCESS);
}

inline void Isend(const void *buf, int count, UMR_Datatype umr_type, int dest, int tag, UMR_Comm comm, UMR_Request *request)
{
  // FPRINTF(stdout, "UMR_Isend rank(w%i) %p[%i] dst(%i) tag(%i)\n", Comm_rank(UMR_COMM_WORLD), buf, count, dest, tag);
  int ret = UMR_Isend(buf, count, umr_type, dest, tag, comm, request);
  assert(ret == UMR_SUCCESS);
}

inline void Wait(UMR_Request *request, UMR_Status *status)
{
  // FPRINTF(stdout, "UMR_Wait rank(w%i)\n", Comm_rank(UMR_COMM_WORLD));
  int ret = UMR_Wait(request, status);
  assert(ret == UMR_SUCCESS);
}

inline bool Test(UMR_Request *request, UMR_Status *status)
{
  int completed = 0;
  // FPRINTF(stdout, "UMR_Test rank(w%i)\n", Comm_rank(UMR_COMM_WORLD));
  int ret = UMR_Test(request, &completed, status);
  assert(ret == UMR_SUCCESS);
  return completed;
}

inline int Waitany(int count, UMR_Request *requests, UMR_Status *status)
{
  int idx = -1;
  // FPRINTF(stdout, "UMR_Waitany rank(w%i) count(%i)\n", Comm_rank(UMR_COMM_WORLD), count);
  int ret = UMR_Waitany(count, requests, &idx, status);
  assert(ret == UMR_SUCCESS);
  return idx;
}

inline int Testany(int count, UMR_Request *requests, UMR_Status *status)
{
  int completed = 0;
  int indx = -1;
  // FPRINTF(stdout, "UMR_Testany rank(w%i) count(%i)\n", Comm_rank(UMR_COMM_WORLD), count);
  int ret = UMR_Testany(count, requests, &indx, &completed, status);
  assert(ret == UMR_SUCCESS);
  return completed ? indx : -1;
}

inline int Waitsome(int incount, UMR_Request *requests, int* indcs, UMR_Status *statuses)
{
  int outcount = 0;
  // FPRINTF(stdout, "UMR_Waitsome rank(w%i) incount(%i)\n", Comm_rank(UMR_COMM_WORLD), incount);
  int ret = UMR_Waitsome(incount, requests, &outcount, indcs, statuses);
  assert(ret == UMR_SUCCESS);
  return outcount;
}

inline int Testsome(int incount, UMR_Request *requests, int* indcs, UMR_Status *statuses)
{
  int outcount = 0;
  // FPRINTF(stdout, "UMR_Testsome rank(w%i) incount(%i)\n", Comm_rank(UMR_COMM_WORLD), incount);
  int ret = UMR_Testsome(incount, requests, &outcount, indcs, statuses);
  assert(ret == UMR_SUCCESS);
  return outcount;
}

inline void Waitall(int count, UMR_Request *requests, UMR_Status *statuses)
{
  // FPRINTF(stdout, "UMR_Waitall rank(w%i) count(%i)\n", Comm_rank(UMR_COMM_WORLD), count);
  int ret = UMR_Waitall(count, requests, statuses);
  assert(ret == UMR_SUCCESS);
}

inline bool Testall(int count, UMR_Request *requests, UMR_Status *statuses)
{
  int completed = 0;
  // FPRINTF(stdout, "UMR_Testall rank(w%i) count(%i)\n", Comm_rank(UMR_COMM_WORLD), count);
  int ret = UMR_Testall(count, requests, &completed, statuses);
  assert(ret == UMR_SUCCESS);
  return completed;
}

} // namespace UMR

} // namespace detail

#endif // COMB_ENABLE_UMR

#endif // _UTILS_UMR_HPP

