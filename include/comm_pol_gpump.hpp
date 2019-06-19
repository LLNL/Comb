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

#ifndef _COMM_POL_GPUMP_HPP
#define _COMM_POL_GPUMP_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_GPUMP

#include "libgpump.h"

#include "for_all.hpp"
#include "utils.hpp"
#include "MessageBase.hpp"

struct gpump_pol {
  // static const bool async = false;
  static const bool mock = false;
  static const char* get_name() { return "gpump"; }
  using communicator_type = struct gpump*;
  static inline communicator_type communicator_create(MPI_Comm comm) { return gpump_init(comm); }
  static inline void communicator_destroy(communicator_type g) { gpump_term(g); }
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


inline void connect_ranks(gpump_pol const&,
                          gpump_pol::communicator_type comm,
                          std::vector<int> const& send_ranks,
                          std::vector<int> const& recv_ranks)
{
  std::set<int> ranks;
  for (int rank : send_ranks) {
    if (ranks.find(rank) != ranks.end()) {
      ranks.insert(rank);
    }
  }
  for (int rank : recv_ranks) {
    if (ranks.find(rank) != ranks.end()) {
      ranks.insert(rank);
    }
  }
  for (int rank : ranks) {
    gpump_connect_propose(comm, rank);
  }
  for (int rank : ranks) {
    gpump_connect_accept(comm, rank);
  }
}

inline void disconnect_ranks(gpump_pol const&,
                             gpump_pol::communicator_type comm,
                             std::vector<int> const& send_ranks,
                             std::vector<int> const& recv_ranks)
{
  std::set<int> ranks;
  for (int rank : send_ranks) {
    if (ranks.find(rank) != ranks.end()) {
      ranks.insert(rank);
    }
  }
  for (int rank : recv_ranks) {
    if (ranks.find(rank) != ranks.end()) {
      ranks.insert(rank);
    }
  }
  for (int rank : ranks) {
    gpump_disconnect(comm, rank);
  }
}


inline void start_send(gpump_pol const&,
                void* buffer, int size, gpump_pol::type_type type,
                int dest_rank, int tag,
                gpump_pol::communicator_type comm, gpump_pol::send_request_type* request)
{
  detail::MPI::Isend(buffer, size, type, dest_rank, tag, comm, request);
}

inline int wait_send_any(gpump_pol const&,
                  int count, gpump_pol::send_request_type* requests,
                  gpump_pol::send_status_type* statuses)
{
  return detail::MPI::Waitany(count, requests, statuses);
}

inline int test_send_any(gpump_pol const&,
                  int count, gpump_pol::send_request_type* requests,
                  gpump_pol::send_status_type* statuses)
{
  return detail::MPI::Testany(count, requests, statuses);
}

inline int wait_send_some(gpump_pol const&,
                   int count, gpump_pol::send_request_type* requests,
                   int* indices, gpump_pol::send_status_type* statuses)
{
  return detail::MPI::Waitsome(count, requests, indices, statuses);
}

inline int test_send_some(gpump_pol const&,
                   int count, gpump_pol::send_request_type* requests,
                   int* indices, gpump_pol::send_status_type* statuses)
{
  return detail::MPI::Testsome(count, requests, indices, statuses);
}

inline void wait_send_all(gpump_pol const&,
                   int count, gpump_pol::send_request_type* requests,
                   gpump_pol::send_status_type* statuses)
{
  detail::MPI::Waitall(count, requests, statuses);
}

inline bool test_send_all(gpump_pol const&,
                   int count, gpump_pol::send_request_type* requests,
                   gpump_pol::send_status_type* statuses)
{
  return detail::MPI::Testall(count, requests, statuses);
}


inline void start_recv(gpump_pol const&,
                void* buffer, int size, gpump_pol::type_type type,
                int src_rank, int tag,
                gpump_pol::communicator_type comm, gpump_pol::send_request_type* request)
{
  detail::MPI::Irecv(buffer, size, type, src_rank, tag, comm, request);
}

inline int wait_recv_any(gpump_pol const&,
                  int count, gpump_pol::recv_request_type* requests,
                  gpump_pol::recv_status_type* statuses)
{
  return detail::MPI::Waitany(count, requests, statuses);
}

inline int test_recv_any(gpump_pol const&,
                  int count, gpump_pol::recv_request_type* requests,
                  gpump_pol::recv_status_type* statuses)
{
  return detail::MPI::Testany(count, requests, statuses);
}

inline int wait_recv_some(gpump_pol const&,
                   int count, gpump_pol::recv_request_type* requests,
                   int* indices, gpump_pol::recv_status_type* statuses)
{
  return detail::MPI::Waitsome(count, requests, indices, statuses);
}

inline int test_recv_some(gpump_pol const&,
                   int count, gpump_pol::recv_request_type* requests,
                   int* indices, gpump_pol::recv_status_type* statuses)
{
  return detail::MPI::Testsome(count, requests, indices, statuses);
}

inline void wait_recv_all(gpump_pol const&,
                   int count, gpump_pol::recv_request_type* requests,
                   gpump_pol::recv_status_type* statuses)
{
  detail::MPI::Waitall(count, requests, statuses);
}

inline bool test_recv_all(gpump_pol const&,
                   int count, gpump_pol::recv_request_type* requests,
                   gpump_pol::recv_status_type* statuses)
{
  return detail::MPI::Testall(count, requests, statuses);
}


template < >
struct Message<gpump_pol> : detail::MessageBase
{
  using policy_comm = gpump_pol;

  using base = detail::MessageBase;

  Message(int partner_rank, int tag, bool have_many)
    : base(partner_rank, tag, have_many)
    , m_region(nullptr)
  { }

  ~Message()
  { }

  template < typename context >
  void pack(context const& con, typename policy_comm::communicator_type)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    DataT* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT const* src = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FPRINTF(stdout, "%p pack %p = %p[%p] len %d\n", this, buf, src, indices, len);
      for_all(con, 0, len, make_copy_idxr_idxr(src, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{}));
      buf += len;
    }
  }

  template < typename context >
  void unpack(context const& con, typename policy_comm::communicator_type)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    DataT const* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT* dst = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FPRINTF(stdout, "%p unpack %p[%p] = %p len %d\n", this, dst, indices, buf, len);
      for_all(con, 0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, dst, detail::indexer_list_idx{indices}));
      buf += len;
    }
  }

  template < typename context >
  void Isend(context const&, typename policy_comm::communicator_type comm, typename policy_comm::send_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    // FPRINTF(stdout, "%p Isend %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
    start_send(policy_comm{}, (void*)m_region/*buffer()*/, nbytes(), 0, partner_rank(), tag(), comm, request);
  }

  template < typename context >
  void Irecv(context const&, typename policy_comm::communicator_type comm, typename policy_comm::recv_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    // FPRINTF(stdout, "%p Irecv %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
    start_recv(policy_comm{}, (void*)m_region/*buffer()*/, nbytes(), 0, partner_rank(), tag(), comm, request);
  }

  template < typename context >
  void allocate(context const&, typename policy_comm::communicator_type comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    if (m_buf == nullptr) {
      m_buf = (DataT*)buf_aloc.allocate(nbytes());
      m_region = gpump_register_region(comm, m_buf, nbytes());
    }
  }

  template < typename context >
  void deallocate(context const&, typename policy_comm::communicator_type comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    if (m_buf != nullptr) {
      gpump_deregister_region(comm, m_region);
      m_region = nullptr;
      buf_aloc.deallocate(m_buf);
      m_buf = nullptr;
    }
  }

private:
  struct ibv_mr* m_region;
};

#endif // COMB_ENABLE_GPUMP

#endif // _COMM_POL_GPUMP_HPP
