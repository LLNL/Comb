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

#ifndef _COMM_POL_MOCK_HPP
#define _COMM_POL_MOCK_HPP

#include "config.hpp"

#include "for_all.hpp"
#include "utils.hpp"

#include <atomic>

struct mock_pol {
  // static const bool async = false;
  static const char* get_name() { return "mock"; }
  using communicator_type = int;
  static inline communicator_type communicator_null() { return 0; }
  using send_request_type = int;
  static inline send_request_type send_request_null() { return 0; }
  using recv_request_type = int;
  static inline recv_request_type recv_request_null() { return 0; }
  using send_status_type = int;
  static inline send_status_type send_status_null() { return 0; }
  using recv_status_type = int;
  static inline recv_status_type recv_status_null() { return 0; }
  using type_type = int;
};


inline void start_send(mock_pol const&,
                void* buffer, int size, mock_pol::type_type type,
                int dest_rank, int tag,
                mock_pol::communicator_type comm, mock_pol::send_request_type* request)
{
  COMB::ignore_unused(buffer, size, type, dest_rank, tag, comm);
  *request = 1;
}

inline int wait_send_any(mock_pol const&,
                  int count, mock_pol::send_request_type* requests,
                  mock_pol::send_status_type* statuses)
{
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
      return i;
    }
  }
  return -1;
}

inline int test_send_any(mock_pol const&,
                  int count, mock_pol::send_request_type* requests,
                  mock_pol::send_status_type* statuses)
{
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
      return i;
    }
  }
  return -1;
}

inline int wait_send_some(mock_pol const&,
                   int count, mock_pol::send_request_type* requests,
                   int* indices, mock_pol::send_status_type* statuses)
{
  int done = 0;
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
      indices[done++] = i;
    }
  }
  return done;
}

inline int test_send_some(mock_pol const&,
                   int count, mock_pol::send_request_type* requests,
                   int* indices, mock_pol::send_status_type* statuses)
{
  int done = 0;
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
      indices[done++] = i;
    }
  }
  return done;
}

inline void wait_send_all(mock_pol const&,
                   int count, mock_pol::send_request_type* requests,
                   mock_pol::send_status_type* statuses)
{
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
    }
  }
}

inline bool test_send_all(mock_pol const&,
                   int count, mock_pol::send_request_type* requests,
                   mock_pol::send_status_type* statuses)
{
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
    }
  }
  return true;
}


inline void start_recv(mock_pol const&,
                void* buffer, int size, mock_pol::type_type type,
                int src_rank, int tag,
                mock_pol::communicator_type comm, mock_pol::send_request_type* request)
{
  COMB::ignore_unused(buffer, size, type, src_rank, tag, comm);
  *request = 1;
}

inline int wait_recv_any(mock_pol const&,
                  int count, mock_pol::recv_request_type* requests,
                  mock_pol::recv_status_type* statuses)
{
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
      return i;
    }
  }
  return -1;
}

inline int test_recv_any(mock_pol const&,
                  int count, mock_pol::recv_request_type* requests,
                  mock_pol::recv_status_type* statuses)
{
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
      return i;
    }
  }
  return -1;
}

inline int wait_recv_some(mock_pol const&,
                   int count, mock_pol::recv_request_type* requests,
                   int* indices, mock_pol::recv_status_type* statuses)
{
  int done = 0;
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
      indices[done++] = i;
    }
  }
  return done;
}

inline int test_recv_some(mock_pol const&,
                   int count, mock_pol::recv_request_type* requests,
                   int* indices, mock_pol::recv_status_type* statuses)
{
  int done = 0;
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
      indices[done++] = i;
    }
  }
  return done;
}

inline void wait_recv_all(mock_pol const&,
                   int count, mock_pol::recv_request_type* requests,
                   mock_pol::recv_status_type* statuses)
{
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
    }
  }
}

inline bool test_recv_all(mock_pol const&,
                   int count, mock_pol::recv_request_type* requests,
                   mock_pol::recv_status_type* statuses)
{
  for (int i = 0; i < count; ++i) {
    if (requests[i] != 2) {
      assert(requests[i] == 1);
      requests[i] = 2;
      statuses[i] = 1;
    }
  }
  return true;
}


#endif // _COMM_POL_MOCK_HPP
