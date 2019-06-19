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

#ifndef _COMM_POL_MPI_HPP
#define _COMM_POL_MPI_HPP

#include "config.hpp"

#include "for_all.hpp"
#include "utils.hpp"
#include "utils_mpi.hpp"

struct mpi_pol {
  // static const bool async = false;
  static const char* get_name() { return "mpi"; }
  using communicator_type = MPI_Comm;
  static inline communicator_type communicator_create(MPI_Comm comm) { return comm; }
  static inline void communicator_destroy(communicator_type) { }
  using send_request_type = MPI_Request;
  static inline send_request_type send_request_null() { return MPI_REQUEST_NULL; }
  using recv_request_type = MPI_Request;
  static inline recv_request_type recv_request_null() { return MPI_REQUEST_NULL; }
  using send_status_type = MPI_Status;
  static inline send_status_type send_status_null() { return send_status_type{}; }
  using recv_status_type = MPI_Status;
  static inline recv_status_type recv_status_null() { return recv_status_type{}; }
  using type_type = MPI_Datatype;
};


inline void connect_ranks(mpi_pol const&,
                          mpi_pol::communicator_type comm,
                          std::vector<int> const& send_ranks,
                          std::vector<int> const& recv_ranks)
{
  COMB::ignore_unused(comm, send_ranks, recv_ranks);
}

inline void disconnect_ranks(mpi_pol const&,
                             mpi_pol::communicator_type comm,
                             std::vector<int> const& send_ranks,
                             std::vector<int> const& recv_ranks)
{
  COMB::ignore_unused(comm, send_ranks, recv_ranks);
}


inline void start_send(mpi_pol const&,
                void* buffer, int size, mpi_pol::type_type type,
                int dest_rank, int tag,
                mpi_pol::communicator_type comm, mpi_pol::send_request_type* request)
{
  detail::MPI::Isend(buffer, size, type, dest_rank, tag, comm, request);
}

inline int wait_send_any(mpi_pol const&,
                  int count, mpi_pol::send_request_type* requests,
                  mpi_pol::send_status_type* statuses)
{
  return detail::MPI::Waitany(count, requests, statuses);
}

inline int test_send_any(mpi_pol const&,
                  int count, mpi_pol::send_request_type* requests,
                  mpi_pol::send_status_type* statuses)
{
  return detail::MPI::Testany(count, requests, statuses);
}

inline int wait_send_some(mpi_pol const&,
                   int count, mpi_pol::send_request_type* requests,
                   int* indices, mpi_pol::send_status_type* statuses)
{
  return detail::MPI::Waitsome(count, requests, indices, statuses);
}

inline int test_send_some(mpi_pol const&,
                   int count, mpi_pol::send_request_type* requests,
                   int* indices, mpi_pol::send_status_type* statuses)
{
  return detail::MPI::Testsome(count, requests, indices, statuses);
}

inline void wait_send_all(mpi_pol const&,
                   int count, mpi_pol::send_request_type* requests,
                   mpi_pol::send_status_type* statuses)
{
  detail::MPI::Waitall(count, requests, statuses);
}

inline bool test_send_all(mpi_pol const&,
                   int count, mpi_pol::send_request_type* requests,
                   mpi_pol::send_status_type* statuses)
{
  return detail::MPI::Testall(count, requests, statuses);
}


inline void start_recv(mpi_pol const&,
                void* buffer, int size, mpi_pol::type_type type,
                int src_rank, int tag,
                mpi_pol::communicator_type comm, mpi_pol::send_request_type* request)
{
  detail::MPI::Irecv(buffer, size, type, src_rank, tag, comm, request);
}

inline int wait_recv_any(mpi_pol const&,
                  int count, mpi_pol::recv_request_type* requests,
                  mpi_pol::recv_status_type* statuses)
{
  return detail::MPI::Waitany(count, requests, statuses);
}

inline int test_recv_any(mpi_pol const&,
                  int count, mpi_pol::recv_request_type* requests,
                  mpi_pol::recv_status_type* statuses)
{
  return detail::MPI::Testany(count, requests, statuses);
}

inline int wait_recv_some(mpi_pol const&,
                   int count, mpi_pol::recv_request_type* requests,
                   int* indices, mpi_pol::recv_status_type* statuses)
{
  return detail::MPI::Waitsome(count, requests, indices, statuses);
}

inline int test_recv_some(mpi_pol const&,
                   int count, mpi_pol::recv_request_type* requests,
                   int* indices, mpi_pol::recv_status_type* statuses)
{
  return detail::MPI::Testsome(count, requests, indices, statuses);
}

inline void wait_recv_all(mpi_pol const&,
                   int count, mpi_pol::recv_request_type* requests,
                   mpi_pol::recv_status_type* statuses)
{
  detail::MPI::Waitall(count, requests, statuses);
}

inline bool test_recv_all(mpi_pol const&,
                   int count, mpi_pol::recv_request_type* requests,
                   mpi_pol::recv_status_type* statuses)
{
  return detail::MPI::Testall(count, requests, statuses);
}


#endif // _COMM_POL_MPI_HPP
